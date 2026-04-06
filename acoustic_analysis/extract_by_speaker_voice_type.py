import argparse
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import AcousticAnalyses_Parselmouth as acoustic

# ==============================
# VS Code 一键运行参数区
# ==============================
# 直接修改这里，然后点击 “Run Python File”。
# input 可以是整个 stimfiles，也可以只指向 SP 或 NV。
QUICK_RUN_ENABLED = True
QUICK_RUN_INPUT = "./stimfiles/NV/" # 
QUICK_RUN_OUTPUT = "./SpeakerVoiceType_Analysis"
QUICK_RUN_SPEAKER = "Sub1"
QUICK_RUN_LIMIT_PER_GROUP = None # 设置为 None 表示不限制数量，设置为具体数字如 5 则每个组只处理前 5 个文件
QUICK_RUN_RENDER_PLOTS = True

# ==============================
# 统一 Pitch 参数设定区
# ==============================
# 以后如果你想统一修改默认的基频提取参数，
# 只需要改这里即可。
# 这个脚本运行时会把这里的设置同步给：
# 1. AcousticAnalyses_Parselmouth.py（声学特征提取）
# 2. plot_spectrum.py（频谱图/音高图）
PITCH_PRESETS = {
    "NV_female": {
        "pitch_floor": 100,
        "pitch_ceiling": 1000,
        "voicing_threshold": 0.50,
        "silence_threshold": 0.09,
        "octave_cost": 0.08,
        "octave_jump_cost": 0.60,
        "voiced_unvoiced_cost": 0.18, 
    },
    "NV_male": {
        "pitch_floor": 60,
        "pitch_ceiling": 700,
        "voicing_threshold": 0.50,
        "silence_threshold": 0.09,
        "octave_cost": 0.08,
        "octave_jump_cost": 0.60,
        "voiced_unvoiced_cost": 0.18,
    },
    "SP_female": {
        "pitch_floor": 100,
        "pitch_ceiling": 800,
        "voicing_threshold": 0.55,
        "silence_threshold": 0.09,
        "octave_cost": 0.10,
        "octave_jump_cost": 0.60,
        "voiced_unvoiced_cost": 0.14,
    },
    "SP_male": {
        "pitch_floor": 50,
        "pitch_ceiling": 500,
        "voicing_threshold": 0.55,
        "silence_threshold": 0.09,
        "octave_cost": 0.10,
        "octave_jump_cost": 0.60,
        "voiced_unvoiced_cost": 0.14,
    },
}




def apply_pitch_presets():
    acoustic.PITCH_PRESETS = {key: dict(value) for key, value in PITCH_PRESETS.items()}


def collect_wav_files(input_dir):
    wav_files = []
    for root, _, files in os.walk(input_dir):
        for file_name in files:
            if file_name.lower().endswith(".wav"):
                wav_files.append(os.path.join(root, file_name))
    wav_files.sort()
    return wav_files


def group_wavs_by_speaker_voice_type(wav_files, input_root, speaker_filter=None):
    grouped = defaultdict(list)
    input_root = os.path.abspath(input_root)

    for wav_path in wav_files:
        meta = acoustic.parse_filename_metadata(wav_path)
        speaker = meta.get("speaker")
        voice_type = meta.get("voice_type")

        if pd.isna(speaker) or pd.isna(voice_type):
            continue

        if speaker_filter and str(speaker) != str(speaker_filter):
            continue

        grouped[(str(speaker), str(voice_type))].append(
            {
                "wav_path": wav_path,
                "relative_path": os.path.relpath(wav_path, input_root),
            }
        )

    return dict(grouped)


def sanitize_group_name(speaker, voice_type):
    return f"{speaker}_{voice_type}"


def ensure_group_dirs(output_root, speaker, voice_type):
    group_name = sanitize_group_name(speaker, voice_type)
    excel_dir = Path(output_root) / "excel"
    plots_dir = Path(output_root) / "plots" / group_name
    excel_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return excel_dir, plots_dir


def process_group(
    speaker,
    voice_type,
    items,
    input_root,
    output_root,
    render_plots=True,
    limit_per_group=None,
):
    import plot_spectrum

    excel_dir, plots_dir = ensure_group_dirs(output_root, speaker, voice_type)
    group_name = sanitize_group_name(speaker, voice_type)

    if limit_per_group is not None:
        items = items[:limit_per_group]

    rows = []
    for item in tqdm(items, desc=f"Processing {group_name}", leave=False):
        wav_path = item["wav_path"]
        relative_path = item["relative_path"]

        try:
            row = acoustic.extract_acoustic_features(wav_path)
            row["RelativePath"] = relative_path
        except Exception as exc:
            row = {
                "FileName": os.path.basename(wav_path),
                "RelativePath": relative_path,
                "speaker": speaker,
                "voice_type": voice_type,
                "ExtractionError": str(exc),
            }

        rows.append(row)

        if render_plots:
            try:
                plot_spectrum.save_diagnostic_plot(wav_path, str(plots_dir))
            except Exception as exc:
                row["PlotError"] = str(exc)

    df = pd.DataFrame(rows)
    excel_path = excel_dir / f"{group_name}_acoustic_features.xlsx"
    df.to_excel(excel_path, index=False)

    return {
        "group": group_name,
        "speaker": speaker,
        "voice_type": voice_type,
        "n_files": len(items),
        "excel_path": str(excel_path),
        "plots_dir": str(plots_dir),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract acoustic features and render diagnostic plots by speaker and voice type."
    )
    parser.add_argument(
        "--input",
        default="./06_Final_VoiceSample",
        help="Root WAV directory. Subdirectories are searched recursively.",
    )
    parser.add_argument(
        "--output",
        default="./SpeakerVoiceType_Analysis",
        help="Root output directory for grouped Excel files and diagnostic plots.",
    )
    parser.add_argument(
        "--speaker",
        default=None,
        help="Optional speaker filter, e.g. Sub1.",
    )
    parser.add_argument(
        "--limit-per-group",
        type=int,
        default=None,
        help="Optional limit for the number of files processed within each speaker-voice_type group.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only extract features; do not render diagnostic plots.",
    )
    return parser.parse_args()


def get_runtime_args():
    if QUICK_RUN_ENABLED:
        return argparse.Namespace(
            input=QUICK_RUN_INPUT,
            output=QUICK_RUN_OUTPUT,
            speaker=QUICK_RUN_SPEAKER,
            limit_per_group=QUICK_RUN_LIMIT_PER_GROUP,
            no_plots=not QUICK_RUN_RENDER_PLOTS,
        )
    return parse_args()


def main():
    args = get_runtime_args()
    apply_pitch_presets()
    input_root = os.path.abspath(args.input)
    output_root = os.path.abspath(args.output)

    wav_files = collect_wav_files(input_root)
    grouped = group_wavs_by_speaker_voice_type(
        wav_files,
        input_root=input_root,
        speaker_filter=args.speaker,
    )

    if not grouped:
        raise ValueError("No matching WAV files were found for the requested grouping/filter.")

    summaries = []
    for (speaker, voice_type), items in tqdm(
        sorted(grouped.items()),
        desc="Speaker-voice_type groups",
    ):
        summary = process_group(
            speaker=speaker,
            voice_type=voice_type,
            items=items,
            input_root=input_root,
            output_root=output_root,
            render_plots=not args.no_plots,
            limit_per_group=args.limit_per_group,
        )
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_path = Path(output_root) / "group_summary.xlsx"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_excel(summary_path, index=False)

    print(f"Saved grouped results to {output_root}")
    print(f"Group summary: {summary_path}")


if __name__ == "__main__":
    main()
