import csv
import sys
import types
from pathlib import Path

import numpy as np
import parselmouth
from core.exporter import PARAMETER_COLUMNS


def _load_analysis_module():
    root_dir = Path(__file__).resolve().parents[1]
    analysis_dir = root_dir / "acoustic_analysis"
    if not analysis_dir.exists():
        analysis_dir = root_dir / "Acoustic_analysis"
    analysis_path = str(analysis_dir)
    if analysis_path not in sys.path:
        sys.path.insert(0, analysis_path)
    if "pandas" not in sys.modules:
        sys.modules["pandas"] = types.SimpleNamespace(DataFrame=None)
    if "tqdm" not in sys.modules:
        sys.modules["tqdm"] = types.SimpleNamespace(tqdm=lambda iterable=None, **kwargs: iterable)
    import AcousticAnalyses_Parselmouth as analysis  # type: ignore
    return analysis


def extract_acoustic_feature_row(audio_path, pitch_params):
    analysis = _load_analysis_module()
    shared_preset = {
        "pitch_floor": float(pitch_params["pitch_floor"]),
        "pitch_ceiling": float(pitch_params["pitch_ceiling"]),
        "voicing_threshold": float(pitch_params["voicing_threshold"]),
        "silence_threshold": float(pitch_params["silence_threshold"]),
        "octave_cost": float(pitch_params["octave_cost"]),
        "octave_jump_cost": float(pitch_params["octave_jump_cost"]),
        "voiced_unvoiced_cost": float(pitch_params["voiced_unvoiced_cost"]),
    }
    analysis.PITCH_PRESETS = {
        "NV_female": dict(shared_preset),
        "NV_male": dict(shared_preset),
        "SP_female": dict(shared_preset),
        "SP_male": dict(shared_preset),
    }
    analysis.ensure_pitch_presets_loaded = lambda: None
    return analysis.extract_acoustic_features(audio_path)


def export_acoustic_features_csv(audio_path, output_path, pitch_params):
    row = extract_acoustic_feature_row(audio_path, pitch_params)
    row["audio_file"] = str(audio_path)
    for key in PARAMETER_COLUMNS:
        row[key] = pitch_params.get(key, "")
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return row


def compute_feature_row_with_pitch_overrides(audio_path, pitch_params, timestamps=None, pitch_values=None, segment_labels=None):
    row = extract_acoustic_feature_row(audio_path, pitch_params)
    row["audio_file"] = str(audio_path)
    for key in PARAMETER_COLUMNS:
        row[key] = pitch_params.get(key, "")
    if timestamps is None or pitch_values is None:
        return row

    timestamps = np.asarray(timestamps, dtype=float)
    pitch_values = np.asarray(pitch_values, dtype=float)
    if len(timestamps) == 0 or len(pitch_values) == 0:
        return row

    if segment_labels is not None:
        segment_labels = np.asarray(segment_labels, dtype=int)
    if segment_labels is None or len(segment_labels) != len(pitch_values):
        segment_labels = np.where(np.isnan(pitch_values), 1, 2).astype(int)

    active_mask = segment_labels != 0
    voiced_mask = active_mask & (~np.isnan(pitch_values)) & (pitch_values > 0)
    active_count = int(np.sum(active_mask))
    voiced_percent = float(np.sum(voiced_mask) / active_count) if active_count > 0 else 0.0

    duration = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
    voiced_binary = voiced_mask.astype(int)
    voiced_segments_count = int(np.sum(np.diff(np.pad(voiced_binary, (1, 0), mode="constant")) == 1))
    voiced_segments_per_sec = voiced_segments_count / duration if duration > 0 else 0.0

    nz_pitch = pitch_values[voiced_mask]
    if len(nz_pitch) > 0:
        f0_semitones = 12.0 * np.log2(nz_pitch / 27.5)
        row.update(
            {
                "Voiced_percent": voiced_percent,
                "VoicedSegmentsPerSec": voiced_segments_per_sec,
                "F0_st_mean": float(np.mean(f0_semitones)),
                "F0_st_median": float(np.median(f0_semitones)),
                "F0_st_SD": float(np.std(f0_semitones)),
                "F0_st_P20": float(np.percentile(f0_semitones, 20)),
                "F0_st_P80": float(np.percentile(f0_semitones, 80)),
                "F0_st_Range_P20_P80": float(np.percentile(f0_semitones, 80) - np.percentile(f0_semitones, 20)),
                "F0_valid_ratio": float(np.mean(~np.isnan(pitch_values))),
            }
        )

        st_track = np.full_like(pitch_values, np.nan, dtype=float)
        st_track[voiced_mask] = 12.0 * np.log2(pitch_values[voiced_mask] / 27.5)
        diffs = np.diff(st_track)
        valid_diffs = diffs[~np.isnan(diffs)]
        if len(valid_diffs) > 0:
            row["F0_Frac_Rise"] = float(np.mean(valid_diffs > 0.25))
            row["F0_Frac_Fall"] = float(np.mean(valid_diffs < -0.25))
        else:
            row["F0_Frac_Rise"] = np.nan
            row["F0_Frac_Fall"] = np.nan
    else:
        row.update(
            {
                "Voiced_percent": voiced_percent,
                "VoicedSegmentsPerSec": voiced_segments_per_sec,
                "F0_st_mean": np.nan,
                "F0_st_median": np.nan,
                "F0_st_SD": np.nan,
                "F0_st_P20": np.nan,
                "F0_st_P80": np.nan,
                "F0_st_Range_P20_P80": np.nan,
                "F0_Frac_Rise": np.nan,
                "F0_Frac_Fall": np.nan,
                "F0_valid_ratio": float(np.mean(~np.isnan(pitch_values))),
            }
        )

    formant_times, f1_values, f2_values, f3_values, bw1_values, bw2_values, bw3_values = compute_formants_for_track(
        audio_path,
        timestamps,
        pitch_values,
        segment_labels,
    )
    if len(f1_values) >= 5:
        row["F1_mean"] = float(np.mean(f1_values))
        row["F2_mean"] = float(np.mean(f2_values))
        row["F3_mean"] = float(np.mean(f3_values))
        row["F1_BW_mean"] = float(np.mean(bw1_values))
        row["F2_BW_mean"] = float(np.mean(bw2_values))
        row["F3_BW_mean"] = float(np.mean(bw3_values))
    else:
        row["F1_mean"] = np.nan
        row["F2_mean"] = np.nan
        row["F3_mean"] = np.nan
        row["F1_BW_mean"] = np.nan
        row["F2_BW_mean"] = np.nan
        row["F3_BW_mean"] = np.nan

    return row


def compute_formants_for_track(audio_path, timestamps, pitch_values, segment_labels=None):
    timestamps = np.asarray(timestamps, dtype=float)
    pitch_values = np.asarray(pitch_values, dtype=float)
    if len(timestamps) == 0 or len(pitch_values) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    if segment_labels is not None and len(segment_labels) == len(pitch_values):
        segment_labels = np.asarray(segment_labels, dtype=int)
        voiced_mask = (segment_labels == 2) & np.isfinite(pitch_values) & (pitch_values > 0)
    else:
        voiced_mask = np.isfinite(pitch_values) & (pitch_values > 0)
    if not np.any(voiced_mask):
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    snd = parselmouth.Sound(str(audio_path))
    max_formant = 5500.0 if "gender2" in str(audio_path).lower() else 5000.0
    try:
        formants = snd.to_formant_burg(
            time_step=0.01,
            max_number_of_formants=5,
            maximum_formant=max_formant,
        )
        voiced_times = timestamps[voiced_mask]
        f1_vals = []
        f2_vals = []
        f3_vals = []
        bw1_vals = []
        bw2_vals = []
        bw3_vals = []
        out_times = []
        for t in voiced_times:
            f1 = formants.get_value_at_time(1, float(t))
            f2 = formants.get_value_at_time(2, float(t))
            f3 = formants.get_value_at_time(3, float(t))
            bw1 = formants.get_bandwidth_at_time(1, float(t))
            bw2 = formants.get_bandwidth_at_time(2, float(t))
            bw3 = formants.get_bandwidth_at_time(3, float(t))
            plausible = (
                not np.isnan(f1)
                and not np.isnan(f2)
                and not np.isnan(f3)
                and not np.isnan(bw1)
                and not np.isnan(bw2)
                and not np.isnan(bw3)
                and 50 < f1 < f2 < f3 < max_formant
                and 0 < bw1 < 1500
                and 0 < bw2 < 2000
                and 0 < bw3 < 2500
            )
            if plausible:
                out_times.append(float(t))
                f1_vals.append(float(f1))
                f2_vals.append(float(f2))
                f3_vals.append(float(f3))
                bw1_vals.append(float(bw1))
                bw2_vals.append(float(bw2))
                bw3_vals.append(float(bw3))
        return (
            np.asarray(out_times, dtype=float),
            np.asarray(f1_vals, dtype=float),
            np.asarray(f2_vals, dtype=float),
            np.asarray(f3_vals, dtype=float),
            np.asarray(bw1_vals, dtype=float),
            np.asarray(bw2_vals, dtype=float),
            np.asarray(bw3_vals, dtype=float),
        )
    except Exception:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
