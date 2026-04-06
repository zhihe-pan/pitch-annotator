import parselmouth
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import argparse
import multiprocessing as mp
from tqdm import tqdm
from matplotlib.patches import Patch
from pathlib import Path
import AcousticAnalyses_Parselmouth as acoustic

_LIBROSA_FOR_PYIN = None

plt.switch_backend("Agg")

# ==============================
# VS Code 一键运行参数区
# ==============================
# 如果你想在 VS Code 里直接点击“Run Python File”来画图，
# 就把 QUICK_RUN_ENABLED 保持为 True，
# 然后只修改下面这几个参数即可。
#
# 常用输入示例：
# "./06_Final_VoiceSample/NV"  -> 只画 NV
# "./06_Final_VoiceSample/SP"  -> 只画 SP
# "./06_Final_VoiceSample"     -> SP 和 NV 都画
#
# 说话者筛选示例：
# "Sub1"    -> 只画 Sub1 的音频
# "Sub10"   -> 只画 Sub10 的音频
# None      -> 不按说话者筛选
#
# limit=10 表示只取前 10 条音频，适合先检查效果。
QUICK_RUN_ENABLED = True
QUICK_RUN_INPUT = "./06_Final_VoiceSample/NV"
QUICK_RUN_OUTPUT = "./tmp_test_plots/NV"
QUICK_RUN_LIMIT = None
QUICK_RUN_WORKERS = 4
QUICK_RUN_SPEAKER = "Sub3"

def get_pitch_settings(file_name):
    return acoustic.get_pitch_settings(file_name, tracking_mode="main")


def get_pitch_bounds(file_name):
    return acoustic.get_pitch_bounds(file_name)


def get_formant_max(file_name):
    return acoustic.get_formant_max(file_name)


def detect_active_intervals(snd):
    return acoustic.detect_active_intervals(snd)


def extract_active_sound(snd, intervals):
    return acoustic.extract_active_sound(snd, intervals)


def build_interval_mask(times, intervals):
    mask = np.zeros(len(times), dtype=bool)
    for st, en in intervals:
        mask |= (times >= st) & (times <= en)
    return mask


def correct_octave_jumps(f0_values):
    corrected, _, _ = acoustic.correct_octave_jumps(f0_values)
    return corrected


def load_librosa_for_pyin():
    """
    Import librosa with a small numba-cache patch.

    In the current acoustics environment (Python 3.12), librosa's pYIN path
    can fail during import because numba caching is enabled for package files.
    We disable that cache path locally here so pYIN can still be used for
    diagnostic plotting.
    """
    global _LIBROSA_FOR_PYIN
    if _LIBROSA_FOR_PYIN is not None:
        return _LIBROSA_FOR_PYIN

    from numba.core import dispatcher, caching
    from numba.np.ufunc import wrappers, ufuncbuilder

    class DummyCache(caching.NullCache): # 
        def __init__(self, *args, **kwargs):
            pass

    class DummyGufCache(wrappers.NullCache):
        def __init__(self, *args, **kwargs):
            pass

    dispatcher.FunctionCache = DummyCache
    ufuncbuilder.FunctionCache = DummyCache
    wrappers.GufWrapperCache = DummyGufCache

    import librosa

    _LIBROSA_FOR_PYIN = librosa
    return _LIBROSA_FOR_PYIN


def remove_short_voiced_runs(values, min_run_frames=3): 
    cleaned = np.array(values, dtype=float)
    voiced_mask = ~np.isnan(cleaned)
    if not np.any(voiced_mask):
        return cleaned

    start = None
    for i, is_voiced in enumerate(voiced_mask):
        if is_voiced and start is None:
            start = i
        elif (not is_voiced) and start is not None:
            if i - start < min_run_frames:
                cleaned[start:i] = np.nan
            start = None

    if start is not None and len(cleaned) - start < min_run_frames:
        cleaned[start:] = np.nan

    return cleaned


def extract_pyin_f0(snd, pitch_floor, pitch_ceiling, time_step=0.01):
    """
    Extract an alternative F0 contour with librosa.pyin on the same silence-removed signal.
    Uses a stable parameter set first, then retries with a denser hop size if
    the first pass yields no valid F0 frames.
    """
    librosa = load_librosa_for_pyin()
    y = snd.values.flatten().astype(float)
    sr = int(round(snd.sampling_frequency))
    if y.size == 0 or sr <= 0:
        return np.array([]), np.array([]), np.array([])

    candidate_configs = [
        {"frame_length": 2048, "step": time_step}, # 默认参数，适合大多数情况
        {"frame_length": 2048, "step": 0.005}, # 更密集的时间步长，适合快速变化的语音
        {"frame_length": 1024, "step": time_step}, # 更短的帧长，适合短语音片段或快速变化的语音
    ]

    floor_hz = float(max(30, pitch_floor)) # 确保最低频率不低于30Hz，避免pyin的底噪问题
    ceiling_hz = float(pitch_ceiling) # pyin的最高频率设置为pitch_ceiling，确保覆盖所有可能的基频范围

    for config in candidate_configs:
        frame_length = config["frame_length"]
        if y.size < frame_length:
            frame_length = 2 ** int(np.floor(np.log2(max(32, y.size)))) # 确保frame_length不超过信号长度，并且至少为32
            frame_length = max(frame_length, 32)
        if frame_length < 32:
            continue

        hop_length = max(1, int(round(sr * config["step"]))) #

        try:
            f0, voiced_flag, _ = librosa.pyin(
                y,
                fmin=floor_hz,
                fmax=ceiling_hz,
                sr=sr,
                frame_length=frame_length,
                hop_length=hop_length,
                beta_parameters=(2, 18), # 增加voicing概率的偏向，减少短暂误判
                resolution=0.05, # 增加频率分辨率，减少八度跳误判
                max_transition_rate=35.92, # 限制频率跳变，减少八度跳误判
                switch_prob=0.01,  # 增加voiced/unvoiced状态切换的平滑度，减少短暂误判
                fill_na=np.nan,
            )
        except Exception:
            continue

        if f0 is None:
            continue

        f0 = np.asarray(f0, dtype=float) # 确保f0是float类型的numpy数组，方便后续处理
        voiced_prob = np.asarray(_, dtype=float) if _ is not None else None #
        if voiced_flag is not None: # 
            voiced_flag = np.asarray(voiced_flag, dtype=bool)
            f0[~voiced_flag] = np.nan # 将非voiced帧的f0设置为NaN，确保后续处理只关注voiced帧
        # if voiced_prob is not None: # 
        #     f0[voiced_prob < 0.30] = np.nan

        f0 = remove_short_voiced_runs(f0, min_run_frames=3)

        valid_count = int(np.sum(~np.isnan(f0)))
        if valid_count == 0:
            continue

        times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
        valid_mask = times <= snd.duration + 1e-6
        if voiced_prob is None:
            voiced_prob = np.full_like(f0, np.nan, dtype=float)
        return times[valid_mask], f0[valid_mask], voiced_prob[valid_mask]

    return np.array([]), np.array([]), np.array([])


def save_diagnostic_plot(
    wav_path,
    output_dir,
    pitch_settings_override=None,
    pitch_postprocess=None,
    title_suffix=None,
):
    snd_raw = parselmouth.Sound(wav_path)
    file_name = os.path.basename(wav_path)
    active_intervals = detect_active_intervals(snd_raw)
    snd = extract_active_sound(snd_raw, active_intervals)
    if snd is None or snd.duration <= 0:
        return

    max_formant = get_formant_max(file_name)
    pitch_settings = pitch_settings_override or get_pitch_settings(file_name)
    min_pitch, max_pitch = get_pitch_bounds(file_name)
    if pitch_settings_override is not None:
        min_pitch = pitch_settings["pitch_floor"]
        max_pitch = pitch_settings["pitch_ceiling"]

    # 2. Use the same active-sound pipeline as the feature extractor.
    pitch = snd.to_pitch_ac(
        time_step=0.01, 
        pitch_floor=pitch_settings["pitch_floor"],
        pitch_ceiling=pitch_settings["pitch_ceiling"],
        voicing_threshold=pitch_settings["voicing_threshold"],
        silence_threshold=pitch_settings["silence_threshold"],
        octave_cost=pitch_settings["octave_cost"],
        octave_jump_cost=pitch_settings["octave_jump_cost"],
        voiced_unvoiced_cost=pitch_settings["voiced_unvoiced_cost"],
    )

    formants = snd.to_formant_burg(
        time_step=0.01, 
        max_number_of_formants=5,
        maximum_formant=max_formant
    )
    intensity = snd.to_intensity(time_step=0.01)
    
    # 提取宽带语谱图
    snd_copy = snd.copy() 
    snd_copy.pre_emphasize(from_frequency=50.0)
    spectrogram = snd_copy.to_spectrogram(
        window_length=0.005, 
        maximum_frequency=max_formant
    )

    # 3. 准备数据坐标
    pitch_times = pitch.xs()
    pitch_values_raw = pitch.selected_array['frequency']
    pitch_values = correct_octave_jumps(pitch_values_raw)
    if pitch_postprocess is not None:
        pitch_values = pitch_postprocess(np.array(pitch_values, dtype=float))
    pyin_times, pyin_values, pyin_probs = extract_pyin_f0(snd, min_pitch, max_pitch, time_step=0.01)
    formant_times = formants.xs()

    max_intensity = np.nanmax(intensity.values)
    micro_silence_threshold = max_intensity - 25.0

    if len(pyin_times) > 0:
        cleaned_pyin = np.array(pyin_values, dtype=float)
        for i, t in enumerate(pyin_times):
            int_val = intensity.get_value(t)
            if math.isnan(int_val) or int_val < micro_silence_threshold:
                cleaned_pyin[i] = np.nan
                continue
            if i < len(pyin_probs) and not np.isnan(pyin_probs[i]) and pyin_probs[i] < 0.70:
                cleaned_pyin[i] = np.nan

        cleaned_pyin = remove_short_voiced_runs(cleaned_pyin, min_run_frames=3)
        pyin_values = cleaned_pyin

    f1_vals, f2_vals, f3_vals = [], [], []
    for t in formant_times:
        pitch_idx = np.searchsorted(pitch_times, t, side='left')
        if pitch_idx >= len(pitch_values):
            pitch_idx = len(pitch_values) - 1
        is_voiced = (
            pitch_idx >= 0 and
            pitch_idx < len(pitch_values) and
            not np.isnan(pitch_values[pitch_idx]) and
            pitch_values[pitch_idx] > 0
        )

        if not is_voiced:
            f1_vals.append(np.nan)
            f2_vals.append(np.nan)
            f3_vals.append(np.nan)
        else:
            f1 = formants.get_value_at_time(1, t)
            f2 = formants.get_value_at_time(2, t)
            f3 = formants.get_value_at_time(3, t)

            plausible = (
                not math.isnan(f1) and not math.isnan(f2) and not math.isnan(f3) and
                50 < f1 < f2 < f3 < max_formant
            )
            f1_vals.append(f1 if plausible else np.nan)
            f2_vals.append(f2 if plausible else np.nan)
            f3_vals.append(f3 if plausible else np.nan)

    status_colors = []
    for i, t in enumerate(pitch_times):
        int_val = intensity.get_value(t)
        p_val = pitch_values[i]

        is_silence = math.isnan(int_val) or int_val < micro_silence_threshold

        if is_silence:
            status_colors.append('white')
        else:
            if not np.isnan(p_val) and p_val > 0:
                status_colors.append('green')
            else:
                status_colors.append('gray')

    valid_pitch = np.where(pitch_values > 0, pitch_values, np.nan)
    mean_f0 = np.nanmean(valid_pitch) 
    mean_f0_str = f"{mean_f0:.1f} Hz" if not np.isnan(mean_f0) else "N/A"
    median_f0 = np.nanmedian(valid_pitch) if np.sum(~np.isnan(valid_pitch)) > 0 else np.nan
    p20_f0 = np.nanpercentile(valid_pitch, 20) if np.sum(~np.isnan(valid_pitch)) > 0 else np.nan
    p80_f0 = np.nanpercentile(valid_pitch, 80) if np.sum(~np.isnan(valid_pitch)) > 0 else np.nan

    # 4. 开始画图
    fig, ax1 = plt.subplots(figsize=(12, 6))
    title = f"Diagnostic Plot: {file_name}  |  Mean F0: {mean_f0_str}"
    if title_suffix:
        title = f"{title}  |  {title_suffix}"
    plt.title(title, fontsize=14, fontweight='bold')

    for i, (t, color) in enumerate(zip(pitch_times, status_colors)):
        if color == 'green':
            ax1.axvspan(t - 0.005, t + 0.005, color=color, alpha=0.15, lw=0)
        elif color == 'gray':
            ax1.axvspan(t - 0.005, t + 0.005, color=color, alpha=0.5, lw=0)

    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_values = np.where(spectrogram.values > 0, spectrogram.values, 1e-10)
    sg_db = 10 * np.log10(sg_values)
    ax1.pcolormesh(X, Y, sg_db, vmin=sg_db.max()-70, vmax=sg_db.max(), cmap='binary', shading='auto', alpha=0.7)

    ax1.scatter(formant_times, f1_vals, c='red', s=8, label='F1', zorder=3)
    ax1.scatter(formant_times, f2_vals, c='darkorange', s=8, label='F2', zorder=3)
    ax1.scatter(formant_times, f3_vals, c='gold', s=8, label='F3', zorder=3)
    
    ax1.set_xlabel("Time (s) [Silence Removed]", fontsize=12)
    ax1.set_ylabel("Spectrogram & Formants (Hz)", fontsize=12, color='black')
    ax1.set_ylim(0, max_formant) 
    
    # 5. 双 Y 轴画基频 (F0)
    ax2 = ax1.twinx()
    
    raw_vals_with_nan = [p if p > 0 else np.nan for p in pitch_values_raw]
    corrected_vals_with_nan = [p if not np.isnan(p) and p > 0 else np.nan for p in pitch_values]

    ax2.plot(
        pitch_times,
        raw_vals_with_nan,
        'o-',
        color='deepskyblue',
        markersize=3,
        linewidth=1.2,
        alpha=0.45,
        label='Raw F0',
        zorder=4,
    )
    ax2.plot(
        pitch_times,
        corrected_vals_with_nan,
        'o-',
        color='blue',
        markersize=4,
        linewidth=2,
        label='Corrected F0',
        zorder=5,
    )
    if len(pyin_times) > 0:
        ax2.plot(
            pyin_times,
            pyin_values,
            linestyle='--',
            color='crimson',
            linewidth=2.2,
            alpha=0.95,
            label='pYIN F0',
            zorder=6,
        )
        pyin_valid = ~np.isnan(pyin_values)
        if np.any(pyin_valid):
            ax2.scatter(
                pyin_times[pyin_valid],
                np.asarray(pyin_values)[pyin_valid],
                marker='x',
                s=18,
                linewidths=0.9,
                color='crimson',
                alpha=0.8,
                zorder=7,
            )
    
    if not np.isnan(mean_f0):
        ax2.axhline(y=mean_f0, color='blue', linestyle='--', linewidth=1.5, alpha=0.6, label='Mean F0')
    if not np.isnan(median_f0):
        ax2.axhline(y=median_f0, color='firebrick', linestyle='-.', linewidth=1.5, alpha=0.8, label='Median F0')
    if not np.isnan(p20_f0):
        ax2.axhline(y=p20_f0, color='royalblue', linestyle=':', linewidth=1.3, alpha=0.8, label='F0 P20')
    if not np.isnan(p80_f0):
        ax2.axhline(y=p80_f0, color='navy', linestyle=':', linewidth=1.3, alpha=0.8, label='F0 P80')

    ax2.set_ylabel("Fundamental Frequency F0 (Hz)", fontsize=12, color='blue')
    ax2.set_ylim(min_pitch, max_pitch * 1.1)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines_1.extend([
        Patch(facecolor='green', alpha=0.15),
        Patch(facecolor='gray', alpha=0.5),
    ])
    labels_1.extend(['Voiced', 'Unvoiced'])
    
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # 6. 保存并释放内存
    out_name = file_name.replace('.wav', '_diag.png')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, out_name), dpi=150, bbox_inches='tight')
    plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate diagnostic spectrogram/F0 plots.")
    parser.add_argument(
        "--input",
        default="./06_Final_VoiceSample",
        help="Input WAV file or directory. If set to the root voice-sample directory, both SP and NV will be processed.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for diagnostic plots",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(6, (os.cpu_count() or 1))),
        help="Number of parallel worker processes for directory mode.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N WAV files in each input directory.",
    )
    parser.add_argument(
        "--speaker",
        default=None,
        help="Only process files whose names start with this speaker ID, e.g. Sub1.",
    )
    return parser.parse_args()


def get_runtime_args():
    if QUICK_RUN_ENABLED:
        return argparse.Namespace(
            input=QUICK_RUN_INPUT,
            output=QUICK_RUN_OUTPUT,
            workers=QUICK_RUN_WORKERS,
            limit=QUICK_RUN_LIMIT,
            speaker=QUICK_RUN_SPEAKER,
        )
    return parse_args()


def infer_output_dir(input_path, output_arg):
    if output_arg:
        return output_arg

    input_path = Path(input_path)
    if input_path.is_file():
        parent_name = input_path.parent.name.upper()
        if parent_name == "SP":
            return "./06_Final_VoiceSample/SP_Diagnostic_Plots/"
        if parent_name == "NV":
            return "./06_Final_VoiceSample/NV_Diagnostic_Plots/"
        return "./Diagnostic_Plots/"

    if input_path.name.upper() == "SP":
        return "./06_Final_VoiceSample/SP_Diagnostic_Plots/"
    if input_path.name.upper() == "NV":
        return "./06_Final_VoiceSample/NV_Diagnostic_Plots/"
    return "./Diagnostic_Plots/"


def _render_one(task):
    wav_path, output_dir = task
    try:
        save_diagnostic_plot(wav_path, output_dir)
        return None
    except Exception as e:
        return f"绘图失败 {os.path.basename(wav_path)}: {e}"


def process_directory(input_dir, output_dir, workers=1, limit=None, speaker=None):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith('.wav'))
    if speaker:
        speaker_lower = speaker.lower()
        files = [f for f in files if f.lower().startswith(speaker_lower + "_")]
    if limit is not None:
        files = files[:limit]
    tasks = [(os.path.join(input_dir, file), output_dir) for file in files]
    errors = []

    if workers <= 1:
        iterator = (_render_one(task) for task in tasks)
        for err in tqdm(iterator, total=len(tasks), desc=f"正在生成诊断图: {Path(input_dir).name}"):
            if err:
                errors.append(err)
    else:
        with mp.Pool(processes=workers) as pool:
            iterator = pool.imap_unordered(_render_one, tasks, chunksize=4)
            for err in tqdm(iterator, total=len(tasks), desc=f"正在生成诊断图: {Path(input_dir).name}"):
                if err:
                    errors.append(err)

    for err in errors:
        print(err)

    print(f"\n全部画完！请前往 {output_dir} 检查。")


def main():
    args = get_runtime_args()
    input_path = Path(args.input)

    if input_path.is_file():
        output_dir = infer_output_dir(input_path, args.output)
        os.makedirs(output_dir, exist_ok=True)
        save_diagnostic_plot(str(input_path), output_dir)
        print(f"\n诊断图已保存到 {output_dir}")
        return

    # If the root voice-sample directory is provided, process SP and NV in one run.
    if input_path.name == "06_Final_VoiceSample":
        for subdir in ["SP", "NV"]:
            sub_input = input_path / subdir
            if sub_input.is_dir():
                sub_output = infer_output_dir(sub_input, None if not args.output else str(Path(args.output) / subdir))
                process_directory(sub_input, sub_output, workers=args.workers, limit=args.limit, speaker=args.speaker)
        return

    output_dir = infer_output_dir(input_path, args.output)
    process_directory(input_path, output_dir, workers=args.workers, limit=args.limit, speaker=args.speaker)


if __name__ == "__main__":
    main()
