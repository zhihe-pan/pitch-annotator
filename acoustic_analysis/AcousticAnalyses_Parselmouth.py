import parselmouth
import pandas as pd
import os
import numpy as np
import argparse
from tqdm import tqdm
from scipy.signal import find_peaks

# Pitch 参数源统一放在 extract_by_speaker_voice_type.py。
# 这个模块只保存运行期同步过来的值。
PITCH_PRESETS = {}


def ensure_pitch_presets_loaded():
    global PITCH_PRESETS
    if PITCH_PRESETS:
        return

    try:
        import extract_by_speaker_voice_type as config_source

        if hasattr(config_source, "PITCH_PRESETS"):
            PITCH_PRESETS = {key: dict(value) for key, value in config_source.PITCH_PRESETS.items()}
            return
    except Exception as exc:
        raise RuntimeError(
            "Pitch presets are not configured. Please edit or run "
            "extract_by_speaker_voice_type.py so the unified pitch presets can be loaded."
        ) from exc

    raise RuntimeError(
        "Pitch presets are not configured. Please edit extract_by_speaker_voice_type.py."
    )


def parse_filename_metadata(filepath):
    file_name = os.path.basename(filepath)
    stem, _ = os.path.splitext(file_name)
    parts = stem.split("_")

    metadata = {
        "speaker": np.nan,
        "emotion": np.nan,
        "voice_type": np.nan,
        "type": np.nan,
    }

    if len(parts) < 6:
        return metadata

    metadata["speaker"] = parts[0]

    try:
        voice_idx = next(i for i, part in enumerate(parts) if part in {"SP", "NV"})
    except StopIteration:
        return metadata

    if voice_idx > 2:
        metadata["emotion"] = "_".join(parts[2:voice_idx])
    metadata["voice_type"] = parts[voice_idx]

    if voice_idx + 1 < len(parts):
        metadata["type"] = parts[voice_idx + 1]

    return metadata


def get_stimulus_type(file_name):
    upper_name = file_name.upper()
    if "_NV_" in upper_name or "/NV/" in upper_name or upper_name.endswith("_NV.WAV"):
        return "NV"
    return "SP"


def get_pitch_settings(file_name, tracking_mode="main"):
    ensure_pitch_presets_loaded()
    is_female = "gender2" in file_name.lower()
    stim_type = get_stimulus_type(file_name)
    preset_key = f"{stim_type}_{'female' if is_female else 'male'}"
    settings = dict(PITCH_PRESETS[preset_key])

    if tracking_mode == "raw":
        settings["octave_cost"] = max(0.08, settings["octave_cost"] - 0.02)
        settings["octave_jump_cost"] = max(0.50, settings["octave_jump_cost"] - 0.15)
        settings["voiced_unvoiced_cost"] = max(0.14, settings["voiced_unvoiced_cost"] - 0.04)

    return settings


def get_pitch_bounds(file_name):
    settings = get_pitch_settings(file_name, tracking_mode="main")
    return settings["pitch_floor"], settings["pitch_ceiling"]


def get_formant_max(file_name):
    is_female = "gender2" in file_name.lower()
    return 5500.0 if is_female else 5000.0


def detect_active_intervals(snd):
    """
    在原始时间轴上检测有效发声段。
    返回 [(start, end), ...]，不在这里拼接，避免扭曲时间动态特征。

    # Active-interval detection adapted from a Praat-based voicing-detection 
    # workflow by Al-Tamimi, with custom thresholding implemented here.
    # Al-Tamimi, J. (2018). Praat-voicing-detection [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.1183876

    """
    try:
        pitch_pass1 = parselmouth.praat.call(
            snd,
            "To Pitch (raw cross-correlation)",
            0.005,
            50,
            1000,
            15,
            "yes",
            0.03,
            0.45,
            0.01,
            0.35,
            0.14,
        ) 
        q5_f0 = parselmouth.praat.call(pitch_pass1, "Get quantile", 0, 0, 0.05, "Hertz")
        if np.isnan(q5_f0) or q5_f0 < 10:
            q5_f0 = 50.0

        intensity_pass2 = parselmouth.praat.call(snd, "To Intensity", q5_f0, 0.005, "yes")
        q5_int = parselmouth.praat.call(intensity_pass2, "Get quantile", 0, 0, 0.05)
        q95_int = parselmouth.praat.call(intensity_pass2, "Get quantile", 0, 0, 0.95)
        int_sd = parselmouth.praat.call(intensity_pass2, "Get standard deviation", 0, 0)
        silence_threshold = -((q95_int - q5_int) - (int_sd / 2))

        tg = parselmouth.praat.call(
            snd,
            "To TextGrid (silences)",
            q5_f0,
            0.005,
            silence_threshold,
            0.1,
            0.1,
            "silent",
            "speech",
        )
        num_intervals = parselmouth.praat.call(tg, "Get number of intervals", 1)
        intervals = []
        for i in range(1, num_intervals + 1):
            label = parselmouth.praat.call(tg, "Get label of interval", 1, i)
            if label == "speech":
                st = parselmouth.praat.call(tg, "Get start time of interval", 1, i)
                en = parselmouth.praat.call(tg, "Get end time of interval", 1, i)
                if en > st:
                    intervals.append((st, en))
        return intervals
    except Exception:
        return [(0.0, snd.duration)]


def extract_active_sound(snd, intervals):
    parts = []
    for st, en in intervals:
        part = parselmouth.praat.call(snd, "Extract part", st, en, "rectangular", 1, "no")
        parts.append(part)

    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return parselmouth.praat.call(parts, "Concatenate")


def build_interval_mask(times, intervals):
    if len(times) == 0 or not intervals:
        return np.zeros(len(times), dtype=bool)
    mask = np.zeros(len(times), dtype=bool)
    for st, en in intervals:
        mask |= (times >= st) & (times <= en)
    return mask


def collect_segmented_peaks(times, values, intervals, distance_frames, prominence):
    count = 0
    for st, en in intervals:
        seg_mask = (times >= st) & (times <= en)
        seg_values = values[seg_mask]
        if len(seg_values) == 0:
            continue
        peaks, _ = find_peaks(seg_values, distance=distance_frames, prominence=prominence)
        count += len(peaks)
    return count


def compute_segmented_rise_fall(times, values, segment_durations, threshold):
    """
    仅在每个拼接活动段内部计算相邻差分，避免拼接边界制造伪 rise/fall。
    """
    if len(values) == 0 or len(times) == 0 or not segment_durations:
        return np.nan, np.nan

    rise_count = 0
    fall_count = 0
    transition_count = 0
    seg_start = 0.0

    for idx, seg_duration in enumerate(segment_durations):
        seg_end = seg_start + seg_duration
        if idx == len(segment_durations) - 1:
            seg_mask = (times >= seg_start) & (times <= seg_end)
        else:
            seg_mask = (times >= seg_start) & (times < seg_end)
        seg_values = np.asarray(values[seg_mask], dtype=float)

        seg_values = seg_values[~np.isnan(seg_values)]
        if len(seg_values) <= 1:
            seg_start = seg_end
            continue

        diffs = np.diff(seg_values)
        transition_count += len(diffs)
        rise_count += np.sum(diffs > threshold)
        fall_count += np.sum(diffs < -threshold)
        seg_start = seg_end

    if transition_count == 0:
        return 0.0, 0.0
    return rise_count / transition_count, fall_count / transition_count


def correct_octave_jumps(f0_values):
    """
    修正常见的 2x / 0.5x 八度跳跃。
    对修正后仍不符合局部连续性的帧，保守地设为 NaN。
    """
    corrected = np.array(f0_values, dtype=float)
    corrected[corrected <= 0] = np.nan
    octave_jump_count = 0

    if len(corrected) < 2:
        valid_ratio = np.mean(~np.isnan(corrected)) if len(corrected) > 0 else 0.0
        return corrected, octave_jump_count, valid_ratio

    for i in range(1, len(corrected)):
        prev = corrected[i - 1]
        cur = corrected[i]
        if np.isnan(prev) or np.isnan(cur) or prev <= 0 or cur <= 0:
            continue

        ratio = cur / prev
        candidate = cur
        changed = False

        if 1.7 <= ratio <= 2.3:
            candidate = cur / 2.0
            changed = True
        elif 0.43 <= ratio <= 0.59:
            candidate = cur * 2.0
            changed = True

        if not changed:
            continue

        local_vals = []
        for j in (i - 2, i - 1, i + 1, i + 2):
            if 0 <= j < len(corrected):
                val = corrected[j]
                if not np.isnan(val) and val > 0:
                    local_vals.append(val)

        baseline = np.median(local_vals) if local_vals else prev
        if baseline > 0:
            original_dev = abs(np.log2(cur / baseline))
            candidate_dev = abs(np.log2(candidate / baseline))
            if candidate_dev + 0.15 < original_dev:
                corrected[i] = candidate
                octave_jump_count += 1
            else:
                corrected[i] = np.nan
        else:
            corrected[i] = np.nan

    valid_ratio = np.mean(~np.isnan(corrected)) if len(corrected) > 0 else 0.0
    return corrected, octave_jump_count, valid_ratio


def extract_acoustic_features(filepath):
    # 加载原始声音
    snd_raw = parselmouth.Sound(filepath)

    file_name = os.path.basename(filepath)
    metadata = parse_filename_metadata(filepath)
    pitch_settings_main = get_pitch_settings(file_name, tracking_mode="main")
    min_pitch, max_pitch = pitch_settings_main["pitch_floor"], pitch_settings_main["pitch_ceiling"]
    max_formant = get_formant_max(file_name)

    active_intervals = detect_active_intervals(snd_raw)
    active_duration = sum(en - st for st, en in active_intervals)
    segment_durations = [en - st for st, en in active_intervals]

    snd = extract_active_sound(snd_raw, active_intervals)

    # 防护：如果剔除静音后什么都没剩下，或者音频太短
    if snd is None or active_duration < 0.05:
        return {'FileName': os.path.basename(filepath)}

    # 此时的 duration 是真正的“有效发声总时长”，不受头尾静音干扰！
    duration = active_duration

    # ================= 1. 基频 (F0 in Semitones) =================
    # 【高亮参数1】这里的 pitch_floor 和 voicing_threshold 直接决定了清浊音判定的死活
    pitch = snd.to_pitch_ac(
        time_step=0.01, 
        pitch_floor=pitch_settings_main["pitch_floor"],
        pitch_ceiling=pitch_settings_main["pitch_ceiling"],
        voicing_threshold=pitch_settings_main["voicing_threshold"],
        silence_threshold=pitch_settings_main["silence_threshold"],
        octave_cost=pitch_settings_main["octave_cost"],
        octave_jump_cost=pitch_settings_main["octave_jump_cost"],
        voiced_unvoiced_cost=pitch_settings_main["voiced_unvoiced_cost"],
    )

    pitch_values_raw = pitch.selected_array['frequency']
    pitch_values, octave_jump_count, f0_valid_ratio = correct_octave_jumps(pitch_values_raw)
    pitch_times = pitch.xs()
    
    # Voiced segments and Voiced % share the same corrected F0 track used for export.
    voiced_indices = (~np.isnan(pitch_values)) & (pitch_values > 0)
    nz_pitch = pitch_values[~np.isnan(pitch_values)]
    voiced_percent = np.mean(voiced_indices) if len(pitch_values) > 0 else 0

    # 计算 VoicedSegmentsPerSec，仅在每个活动段内部计数，避免拼接制造假连续
    voiced_segments_count = 0
    seg_start = 0.0
    for idx, seg_duration in enumerate(segment_durations):
        seg_end = seg_start + seg_duration
        if idx == len(segment_durations) - 1:
            seg_mask = (pitch_times >= seg_start) & (pitch_times <= seg_end)
        else:
            seg_mask = (pitch_times >= seg_start) & (pitch_times < seg_end)
        seg_voiced = voiced_indices[seg_mask].astype(int)
        if len(seg_voiced) == 0:
            seg_start = seg_end
            continue
        voiced_segments_count += np.sum(np.diff(np.pad(seg_voiced, (1, 0), mode='constant')) == 1)
        seg_start = seg_end
    voiced_seg_per_sec = voiced_segments_count / duration if duration > 0 else 0

    if len(nz_pitch) > 0:
        f0_semitones = 12 * np.log2(nz_pitch / 27.5)
        f0_semitones_track = np.full_like(pitch_values, np.nan, dtype=float)
        valid_pitch_mask = ~np.isnan(pitch_values)
        f0_semitones_track[valid_pitch_mask] = 12 * np.log2(pitch_values[valid_pitch_mask] / 27.5)
        
        f0_st_mean = np.mean(f0_semitones)
        f0_st_median = np.median(f0_semitones)
        f0_st_min = np.min(f0_semitones)
        f0_st_max = np.max(f0_semitones)
        f0_st_sd = np.std(f0_semitones)
        f0_st_range = f0_st_max - f0_st_min
        
        # 分位数计算
        f0_st_p20 = np.percentile(f0_semitones, 20)  
        f0_st_p80 = np.percentile(f0_semitones, 80)  
        f0_st_p20 = np.percentile(f0_semitones, 20)
        f0_st_p80 = np.percentile(f0_semitones, 80)
        f0_st_range_20_80 = f0_st_p80 - f0_st_p20
        
        # F0 动态变化比例 (Frac Rise / Fall)，仅在活动段内部统计
        st_diff_thresh = 0.25
        f0_frac_rise, f0_frac_fall = compute_segmented_rise_fall(
            pitch_times,
            f0_semitones_track,
            segment_durations,
            st_diff_thresh,
        )
    else:
        f0_st_mean = f0_st_median = f0_st_min = f0_st_max = f0_st_sd = f0_st_range = np.nan
        f0_st_p20 = f0_st_p80 = f0_st_range_20_80 = np.nan
        f0_frac_rise = f0_frac_fall = np.nan

    # ================= 2. 强度/响度 (Intensity) =================
    intensity_raw = snd_raw.to_intensity(time_step=0.01)
    intensity_raw_values = intensity_raw.values.flatten()
    intensity_raw_times = intensity_raw.xs()
    active_int_mask = build_interval_mask(intensity_raw_times, active_intervals)

    intensity = snd.to_intensity(time_step=0.01)
    intensity_times = intensity.xs()
    int_values = intensity.values.flatten()
    valid_int = int_values[int_values > 0]
    int_track = np.where(int_values > 0, int_values, np.nan)
    
    # 提取响度峰值频率 (loudnessPeaksPerSec)
    loudness_peak_count = collect_segmented_peaks(
        intensity_raw_times,
        intensity_raw_values,
        active_intervals,
        distance_frames=10,
        prominence=3,
    )
    loudness_peaks_per_sec = loudness_peak_count / duration if duration > 0 else 0

    if len(valid_int) > 0:
        int_mean = np.mean(valid_int)
        int_median = np.median(valid_int)
        int_min = np.min(valid_int)
        int_max = np.max(valid_int)
        int_sd = np.std(valid_int)
        int_range = int_max - int_min
        
        int_p20 = np.percentile(valid_int, 20)
        int_p80 = np.percentile(valid_int, 80)
        int_p20 = np.percentile(valid_int, 20)
        int_p80 = np.percentile(valid_int, 80)
        int_range_20_80 = int_p80 - int_p20
        
        int_diff_thresh = 1.0
        int_frac_rise, int_frac_fall = compute_segmented_rise_fall(
            intensity_times,
            int_track,
            segment_durations,
            int_diff_thresh,
        )
    else:
        int_mean = int_median = int_min = int_max = int_sd = int_range = np.nan
        int_p20 = int_p80 = int_range_20_80 = np.nan
        int_frac_rise = int_frac_fall = np.nan

    # ================= 3. 音质 (Jitter, Shimmer, HNR) =================
    try:
        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", min_pitch, max_pitch)
        jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    except:
        jitter, shimmer = np.nan, np.nan

    try:
        harmonicity = snd.to_harmonicity_cc()
        hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
    except:
        hnr = np.nan

    # ================= 4. 频谱特征 (Spectrum) =================
    try:
        spectrum = snd.to_spectrum()
        cog = spectrum.get_center_of_gravity(2.0)
        
        total_energy = parselmouth.praat.call(spectrum, "Get band energy", 0, 0) 
        energy_above_500 = parselmouth.praat.call(spectrum, "Get band energy", 500, 0)
        energy_above_1000 = parselmouth.praat.call(spectrum, "Get band energy", 1000, 0)
        energy_below_500 = parselmouth.praat.call(spectrum, "Get band energy", 0, 500)
        energy_below_1000 = parselmouth.praat.call(spectrum, "Get band energy", 0, 1000)
        
        hf500_ratio = energy_above_500 / energy_below_500 if energy_below_500 > 0 else np.nan
        hf1000_ratio = energy_above_1000 / energy_below_1000 if energy_below_1000 > 0 else np.nan
        
        bin_frequencies = spectrum.xs()
        bin_power = spectrum.values[0]**2 + spectrum.values[1]**2
        valid_bins = bin_power > 0
        if np.sum(valid_bins) > 1:
            freqs_valid = bin_frequencies[valid_bins]
            power_db = 10 * np.log10(bin_power[valid_bins])
            slope, _ = np.polyfit(freqs_valid, power_db, 1)
            spectrum_slope = slope
        else:
            spectrum_slope = np.nan
    except:
        cog, hf500_ratio, hf1000_ratio, spectrum_slope = np.nan, np.nan, np.nan, np.nan

    # ================= 5. 共振峰 (Formants) =================
    try:
        formant = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=max_formant)
        f1_list, f2_list, f3_list = [], [], []
        f1_bw_list, f2_bw_list, f3_bw_list = [], [], []
        formant_pitch = snd.to_pitch_ac(
            time_step=0.01,
            pitch_floor=pitch_settings_main["pitch_floor"],
            pitch_ceiling=pitch_settings_main["pitch_ceiling"],
            voicing_threshold=pitch_settings_main["voicing_threshold"],
            silence_threshold=pitch_settings_main["silence_threshold"],
            octave_cost=pitch_settings_main["octave_cost"],
            octave_jump_cost=pitch_settings_main["octave_jump_cost"],
            voiced_unvoiced_cost=pitch_settings_main["voiced_unvoiced_cost"],
        )
        formant_pitch_times = formant_pitch.xs()
        formant_pitch_values, _, _ = correct_octave_jumps(formant_pitch.selected_array['frequency'])
        valid_formant_frames = 0
        
        for t in np.arange(0, snd.duration, 0.01):
            pitch_idx = np.searchsorted(formant_pitch_times, t, side='left')
            if pitch_idx >= len(formant_pitch_values):
                pitch_idx = len(formant_pitch_values) - 1
            is_voiced_frame = (
                pitch_idx >= 0 and
                pitch_idx < len(formant_pitch_values) and
                not np.isnan(formant_pitch_values[pitch_idx]) and
                formant_pitch_values[pitch_idx] > 0
            )

            if is_voiced_frame:
                f1 = formant.get_value_at_time(1, t)
                f2 = formant.get_value_at_time(2, t)
                f3 = formant.get_value_at_time(3, t)
                bw1 = formant.get_bandwidth_at_time(1, t)
                bw2 = formant.get_bandwidth_at_time(2, t)
                bw3 = formant.get_bandwidth_at_time(3, t)

                is_plausible = (
                    not np.isnan(f1) and not np.isnan(f2) and not np.isnan(f3) and
                    50 < f1 < f2 < f3 < max_formant and
                    0 < bw1 < 1500 and 0 < bw2 < 2000 and 0 < bw3 < 2500
                )
                if is_plausible:
                    valid_formant_frames += 1
                    f1_list.append(f1)
                    f2_list.append(f2)
                    f3_list.append(f3)
                    f1_bw_list.append(bw1)
                    f2_bw_list.append(bw2)
                    f3_bw_list.append(bw3)
                
        if valid_formant_frames >= 5:
            f1_mean = np.mean(f1_list) if f1_list else np.nan
            f2_mean = np.mean(f2_list) if f2_list else np.nan
            f3_mean = np.mean(f3_list) if f3_list else np.nan
            f1_bw = np.mean(f1_bw_list) if f1_bw_list else np.nan
            f2_bw = np.mean(f2_bw_list) if f2_bw_list else np.nan
            f3_bw = np.mean(f3_bw_list) if f3_bw_list else np.nan
        else:
            f1_mean, f2_mean, f3_mean, f1_bw, f2_bw, f3_bw = [np.nan] * 6
    except:
        f1_mean, f2_mean, f3_mean, f1_bw, f2_bw, f3_bw = [np.nan]*6

    # ================= 6. 返回 =================
    return {
        'FileName': os.path.basename(filepath),
        'speaker': metadata["speaker"],
        'emotion': metadata["emotion"],
        'voice_type': metadata["voice_type"],
        'type': metadata["type"],
        'Voiced_percent': voiced_percent,
        'loudnessPeaksPerSec': loudness_peaks_per_sec,
        'VoicedSegmentsPerSec': voiced_seg_per_sec,
        
        'Int_mean': int_mean,
        'Int_median': int_median,
        'Int_SD': int_sd,
        'Int_P20': int_p20,
        'Int_P80': int_p80,
        'Int_Range_P20_P80': int_range_20_80,
        'Int_Frac_Rise': int_frac_rise, 'Int_Frac_Fall': int_frac_fall,
        'Shimmer': shimmer, 'HNR_dB': hnr,
        
        'F0_st_mean': f0_st_mean,
        'F0_st_median': f0_st_median,
        'F0_st_SD': f0_st_sd,
        'F0_st_P20': f0_st_p20,
        'F0_st_P80': f0_st_p80,
        'F0_st_Range_P20_P80': f0_st_range_20_80,
        'F0_Frac_Rise': f0_frac_rise, 'F0_Frac_Fall': f0_frac_fall,
        'F0_valid_ratio': f0_valid_ratio,
        'F0_octave_jump_count': octave_jump_count,
        'Jitter': jitter,
        
        'F1_mean': f1_mean, 'F2_mean': f2_mean, 'F3_mean': f3_mean,
        'F1_BW_mean': f1_bw, 'F2_BW_mean': f2_bw, 'F3_BW_mean': f3_bw,
        
        'COG_Hz': cog, 'HF500_ratio': hf500_ratio, 'HF1000_ratio': hf1000_ratio,
        'Spectrum_slope': spectrum_slope
    }


def extract_features_from_directory(input_dir):
    input_dir = os.path.abspath(input_dir)
    wav_files = []
    for root, _, files in os.walk(input_dir):
        for file_name in files:
            if file_name.lower().endswith(".wav"):
                wav_files.append(os.path.join(root, file_name))

    wav_files.sort()
    results = []
    for wav_path in tqdm(wav_files, desc="Extracting acoustic features"):
        try:
            row = extract_acoustic_features(wav_path)
            row["RelativePath"] = os.path.relpath(wav_path, input_dir)
            results.append(row)
        except Exception as exc:
            results.append(
                {
                    "FileName": os.path.basename(wav_path),
                    "RelativePath": os.path.relpath(wav_path, input_dir),
                    "ExtractionError": str(exc),
                }
            )

    return pd.DataFrame(results)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract acoustic features from WAV files and save to Excel.")
    parser.add_argument(
        "--input",
        default="./06_Final_VoiceSample",
        help="Input WAV directory. Subdirectories are searched recursively.",
    )
    parser.add_argument(
        "--output",
        default="AcousticFeatures_Parselmouth.xlsx",
        help="Output Excel file path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = extract_features_from_directory(args.input)
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
