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


def _estimate_frame_half_width(timestamps):
    timestamps = np.asarray(timestamps, dtype=float)
    if len(timestamps) > 1:
        return max(float(np.median(np.diff(timestamps))) / 2.0, 1e-4)
    return 0.005


def _build_active_intervals_from_labels(timestamps, segment_labels):
    timestamps = np.asarray(timestamps, dtype=float)
    segment_labels = np.asarray(segment_labels, dtype=int)
    if len(timestamps) == 0 or len(segment_labels) == 0 or len(timestamps) != len(segment_labels):
        return []

    half_width = _estimate_frame_half_width(timestamps)
    active_mask = segment_labels != 0
    if not np.any(active_mask):
        return []

    intervals = []
    start_idx = None
    for idx, is_active in enumerate(active_mask):
        if is_active and start_idx is None:
            start_idx = idx
        elif (not is_active) and start_idx is not None:
            start = max(0.0, float(timestamps[start_idx] - half_width))
            end = float(timestamps[idx - 1] + half_width)
            if end > start:
                intervals.append((start, end))
            start_idx = None

    if start_idx is not None:
        start = max(0.0, float(timestamps[start_idx] - half_width))
        end = float(timestamps[-1] + half_width)
        if end > start:
            intervals.append((start, end))

    return intervals


def _interval_mask(times, start, end, is_last):
    times = np.asarray(times, dtype=float)
    if is_last:
        return (times >= start) & (times <= end)
    return (times >= start) & (times < end)


def _compute_segmented_rise_fall_for_track(times, values, intervals, threshold):
    if len(times) == 0 or len(values) == 0 or not intervals:
        return np.nan, np.nan

    rise_count = 0
    fall_count = 0
    transition_count = 0
    for idx, (start, end) in enumerate(intervals):
        seg_mask = _interval_mask(times, start, end, idx == len(intervals) - 1)
        seg_values = np.asarray(values[seg_mask], dtype=float)
        seg_values = seg_values[~np.isnan(seg_values)]
        if len(seg_values) <= 1:
            continue
        diffs = np.diff(seg_values)
        transition_count += len(diffs)
        rise_count += int(np.sum(diffs > threshold))
        fall_count += int(np.sum(diffs < -threshold))

    if transition_count == 0:
        return 0.0, 0.0
    return rise_count / transition_count, fall_count / transition_count


def _empty_active_override_row():
    return {
        "loudnessPeaksPerSec": 0.0,
        "Int_mean": np.nan,
        "Int_median": np.nan,
        "Int_SD": np.nan,
        "Int_P20": np.nan,
        "Int_P80": np.nan,
        "Int_Range_P20_P80": np.nan,
        "Int_Frac_Rise": np.nan,
        "Int_Frac_Fall": np.nan,
        "Jitter": np.nan,
        "Shimmer": np.nan,
        "HNR_dB": np.nan,
        "COG_Hz": np.nan,
        "HF500_ratio": np.nan,
        "HF1000_ratio": np.nan,
        "Spectrum_slope": np.nan,
    }


def _compute_activity_dependent_metrics(audio_path, active_intervals, pitch_params):
    if not active_intervals:
        return _empty_active_override_row()

    active_duration = float(sum(end - start for start, end in active_intervals))
    if active_duration < 0.05:
        return _empty_active_override_row()

    analysis = _load_analysis_module()
    snd_raw = parselmouth.Sound(str(audio_path))
    snd = analysis.extract_active_sound(snd_raw, active_intervals)
    if snd is None or snd.duration <= 0:
        return _empty_active_override_row()

    segment_durations = [end - start for start, end in active_intervals]

    intensity_raw = snd_raw.to_intensity(time_step=0.01)
    intensity_raw_values = intensity_raw.values.flatten()
    intensity_raw_times = intensity_raw.xs()
    loudness_peak_count = analysis.collect_segmented_peaks(
        intensity_raw_times,
        intensity_raw_values,
        active_intervals,
        distance_frames=10,
        prominence=3,
    )
    loudness_peaks_per_sec = loudness_peak_count / active_duration if active_duration > 0 else 0.0

    intensity = snd.to_intensity(time_step=0.01)
    intensity_times = intensity.xs()
    int_values = intensity.values.flatten()
    valid_int = int_values[int_values > 0]
    int_track = np.where(int_values > 0, int_values, np.nan)
    if len(valid_int) > 0:
        int_mean = float(np.mean(valid_int))
        int_median = float(np.median(valid_int))
        int_sd = float(np.std(valid_int))
        int_p20 = float(np.percentile(valid_int, 20))
        int_p80 = float(np.percentile(valid_int, 80))
        int_range_20_80 = float(int_p80 - int_p20)
        int_frac_rise, int_frac_fall = analysis.compute_segmented_rise_fall(
            intensity_times,
            int_track,
            segment_durations,
            1.0,
        )
    else:
        int_mean = int_median = int_sd = int_p20 = int_p80 = int_range_20_80 = np.nan
        int_frac_rise = int_frac_fall = np.nan

    min_pitch = float(pitch_params["pitch_floor"])
    max_pitch = float(pitch_params["pitch_ceiling"])
    try:
        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", min_pitch, max_pitch)
        jitter = float(parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
        shimmer = float(parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
    except Exception:
        jitter, shimmer = np.nan, np.nan

    try:
        harmonicity = snd.to_harmonicity_cc()
        hnr = float(parselmouth.praat.call(harmonicity, "Get mean", 0, 0))
    except Exception:
        hnr = np.nan

    try:
        spectrum = snd.to_spectrum()
        cog = float(spectrum.get_center_of_gravity(2.0))
        energy_above_500 = parselmouth.praat.call(spectrum, "Get band energy", 500, 0)
        energy_above_1000 = parselmouth.praat.call(spectrum, "Get band energy", 1000, 0)
        energy_below_500 = parselmouth.praat.call(spectrum, "Get band energy", 0, 500)
        energy_below_1000 = parselmouth.praat.call(spectrum, "Get band energy", 0, 1000)
        hf500_ratio = float(energy_above_500 / energy_below_500) if energy_below_500 > 0 else np.nan
        hf1000_ratio = float(energy_above_1000 / energy_below_1000) if energy_below_1000 > 0 else np.nan

        bin_frequencies = spectrum.xs()
        bin_power = spectrum.values[0] ** 2 + spectrum.values[1] ** 2
        valid_bins = bin_power > 0
        if np.sum(valid_bins) > 1:
            freqs_valid = bin_frequencies[valid_bins]
            power_db = 10.0 * np.log10(bin_power[valid_bins])
            slope, _ = np.polyfit(freqs_valid, power_db, 1)
            spectrum_slope = float(slope)
        else:
            spectrum_slope = np.nan
    except Exception:
        cog, hf500_ratio, hf1000_ratio, spectrum_slope = np.nan, np.nan, np.nan, np.nan

    return {
        "loudnessPeaksPerSec": float(loudness_peaks_per_sec),
        "Int_mean": int_mean,
        "Int_median": int_median,
        "Int_SD": int_sd,
        "Int_P20": int_p20,
        "Int_P80": int_p80,
        "Int_Range_P20_P80": int_range_20_80,
        "Int_Frac_Rise": float(int_frac_rise) if not np.isnan(int_frac_rise) else np.nan,
        "Int_Frac_Fall": float(int_frac_fall) if not np.isnan(int_frac_fall) else np.nan,
        "Jitter": jitter,
        "Shimmer": shimmer,
        "HNR_dB": hnr,
        "COG_Hz": cog,
        "HF500_ratio": hf500_ratio,
        "HF1000_ratio": hf1000_ratio,
        "Spectrum_slope": spectrum_slope,
    }


def _project_track_to_active_timeline(timestamps, pitch_values, segment_labels, active_intervals):
    timestamps = np.asarray(timestamps, dtype=float)
    pitch_values = np.asarray(pitch_values, dtype=float)
    segment_labels = np.asarray(segment_labels, dtype=int)
    if len(timestamps) == 0 or len(pitch_values) == 0 or len(segment_labels) != len(pitch_values):
        return np.array([]), np.array([]), np.array([], dtype=int)
    if not active_intervals:
        return np.array([]), np.array([]), np.array([], dtype=int)

    projected_times = []
    projected_pitch = []
    projected_labels = []
    offset = 0.0
    for idx, (start, end) in enumerate(active_intervals):
        seg_mask = _interval_mask(timestamps, start, end, idx == len(active_intervals) - 1)
        if not np.any(seg_mask):
            offset += end - start
            continue
        seg_times = timestamps[seg_mask]
        projected_times.append(seg_times - float(start) + offset)
        projected_pitch.append(pitch_values[seg_mask])
        projected_labels.append(segment_labels[seg_mask])
        offset += end - start

    if not projected_times:
        return np.array([]), np.array([]), np.array([], dtype=int)

    return (
        np.concatenate(projected_times).astype(float, copy=False),
        np.concatenate(projected_pitch).astype(float, copy=False),
        np.concatenate(projected_labels).astype(int, copy=False),
    )


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
    active_intervals = _build_active_intervals_from_labels(timestamps, segment_labels)
    active_times, active_pitch_values, _ = _project_track_to_active_timeline(
        timestamps,
        pitch_values,
        segment_labels,
        active_intervals,
    )
    analysis = _load_analysis_module()
    corrected_active_pitch, octave_jump_count, _ = analysis.correct_octave_jumps(active_pitch_values)
    row["F0_octave_jump_count"] = int(octave_jump_count)
    active_count = int(np.sum(active_mask))
    voiced_percent = float(np.sum(voiced_mask) / active_count) if active_count > 0 else 0.0
    row.update(_compute_activity_dependent_metrics(audio_path, active_intervals, pitch_params))

    duration = float(sum(end - start for start, end in active_intervals))
    voiced_segments_count = 0
    for idx, (start, end) in enumerate(active_intervals):
        seg_mask = _interval_mask(timestamps, start, end, idx == len(active_intervals) - 1)
        seg_voiced = voiced_mask[seg_mask].astype(int)
        if len(seg_voiced) == 0:
            continue
        voiced_segments_count += int(np.sum(np.diff(np.pad(seg_voiced, (1, 0), mode="constant")) == 1))
    voiced_segments_per_sec = voiced_segments_count / duration if duration > 0 else 0.0

    nz_pitch = pitch_values[voiced_mask]
    if len(nz_pitch) > 0:
        f0_semitones = 12.0 * np.log2(nz_pitch / 27.5)
        st_track = np.full_like(pitch_values, np.nan, dtype=float)
        st_track[voiced_mask] = 12.0 * np.log2(pitch_values[voiced_mask] / 27.5)
        frac_rise, frac_fall = _compute_segmented_rise_fall_for_track(
            timestamps,
            st_track,
            active_intervals,
            0.25,
        )
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
                "F0_valid_ratio": float(np.mean((~np.isnan(pitch_values[active_mask])) & (pitch_values[active_mask] > 0))) if active_count > 0 else 0.0,
                "F0_Frac_Rise": frac_rise,
                "F0_Frac_Fall": frac_fall,
            }
        )
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
                "F0_valid_ratio": float(np.mean((~np.isnan(pitch_values[active_mask])) & (pitch_values[active_mask] > 0))) if active_count > 0 else 0.0,
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

    if segment_labels is None or len(segment_labels) != len(pitch_values):
        segment_labels = np.where(np.isnan(pitch_values), 1, 2).astype(int)
    else:
        segment_labels = np.asarray(segment_labels, dtype=int)

    active_intervals = _build_active_intervals_from_labels(timestamps, segment_labels)
    if not active_intervals:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    analysis = _load_analysis_module()
    snd_raw = parselmouth.Sound(str(audio_path))
    snd = analysis.extract_active_sound(snd_raw, active_intervals)
    if snd is None or snd.duration <= 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    active_times, active_pitch_values, active_segment_labels = _project_track_to_active_timeline(
        timestamps,
        pitch_values,
        segment_labels,
        active_intervals,
    )
    voiced_mask = (active_segment_labels == 2) & np.isfinite(active_pitch_values) & (active_pitch_values > 0)
    if not np.any(voiced_mask):
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    max_formant = 5500.0 if "gender2" in str(audio_path).lower() else 5000.0
    try:
        formants = snd.to_formant_burg(
            time_step=0.01,
            max_number_of_formants=5,
            maximum_formant=max_formant,
        )
        f1_vals = []
        f2_vals = []
        f3_vals = []
        bw1_vals = []
        bw2_vals = []
        bw3_vals = []
        out_times = []
        frame_step = 0.01
        frame_times = np.arange(0.0, snd.duration, frame_step, dtype=float)
        if len(frame_times) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        valid_active = np.isfinite(active_pitch_values)
        if not np.any(valid_active):
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        for t in frame_times:
            pitch_idx = int(np.searchsorted(active_times, t, side="left"))
            if pitch_idx >= len(active_pitch_values):
                pitch_idx = len(active_pitch_values) - 1
            if pitch_idx < 0:
                continue
            is_voiced_frame = (
                active_segment_labels[pitch_idx] == 2
                and np.isfinite(active_pitch_values[pitch_idx])
                and active_pitch_values[pitch_idx] > 0
            )
            if not is_voiced_frame:
                continue
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
