import numpy as np
import librosa
import parselmouth
from scipy.signal import find_peaks

SEGMENT_SILENCE = 0
SEGMENT_VOICELESS = 1
SEGMENT_VOICED = 2

class AudioProcessor:
    def __init__(self):
        self.audio_data = None
        self.sr = None
        self.loaded_filepath = None
        
        # Spectrogram data
        self.S_db = None
        self.spec_times = None
        self.spec_freqs = None
        self.window_length = 0.005
        self.maximum_frequency = 5000.0
        self.dynamic_range_db = 70.0
        self.filtered_ac_attenuation_at_top = 0.03
    
    def _maximum_formant_for_file(self):
        filepath = "" if self.loaded_filepath is None else str(self.loaded_filepath).lower()
        return 5500.0 if "gender2" in filepath else 5000.0

    def load_audio(self, filepath):
        """
        Load audio using librosa. It handles most standard formats.
        """
        y, sr = librosa.load(filepath, sr=None)
        
        # Convert to mono if it's stereo
        if y.ndim > 1:
            y = librosa.to_mono(y)
            
        self.audio_data = y
        self.sr = sr
        self.loaded_filepath = str(filepath)
        self._compute_spectrogram()
        
        return self.audio_data, self.sr

    def _compute_spectrogram(self):
        """
        Compute a Praat-like spectrogram through Parselmouth so the display
        matches Praat's own analysis more closely.
        """
        if self.audio_data is None:
            return

        snd = parselmouth.Sound(self.audio_data, self.sr)
        snd_for_spec = snd.copy()
        snd_for_spec.pre_emphasize(from_frequency=50.0)
        max_frequency = min(self._maximum_formant_for_file(), self.sr / 2.0)
        self.maximum_frequency = float(max_frequency)
        spectrogram = snd_for_spec.to_spectrogram(
            window_length=self.window_length,
            maximum_frequency=max_frequency,
        )

        power = np.maximum(spectrogram.values, np.finfo(float).tiny)
        self.S_db = 10.0 * np.log10(power)
        self.spec_times = spectrogram.xs()
        self.spec_freqs = spectrogram.ys()

    def extract_pitch(
        self,
        pitch_floor=50.0,
        pitch_ceiling=800.0,
        time_step=0.0,
        voicing_threshold=0.50,
        silence_threshold=0.09,
        octave_cost=0.055,
        octave_jump_cost=0.35,
        voiced_unvoiced_cost=0.14,
    ):
        """
        Extract pitch using parselmouth
        """
        if self.audio_data is None:
            return np.array([]), np.array([]), np.array([], dtype=int), np.array([]), np.array([]), np.array([]), np.array([])

        raw_snd = parselmouth.Sound(self.audio_data, self.sr)
        filtered_audio = self._apply_filtered_ac_lowpass(
            self.audio_data,
            self.sr,
            pitch_ceiling,
            self.filtered_ac_attenuation_at_top,
        )
        snd = parselmouth.Sound(filtered_audio, self.sr)
        resolved_time_step = None if time_step <= 0 else time_step
        pitch = snd.to_pitch_ac(
            time_step=resolved_time_step,
            pitch_floor=pitch_floor,
            pitch_ceiling=pitch_ceiling,
            voicing_threshold=voicing_threshold,
            silence_threshold=silence_threshold,
            octave_cost=octave_cost,
            octave_jump_cost=octave_jump_cost,
            voiced_unvoiced_cost=voiced_unvoiced_cost,
        )
        
        pitch_values = pitch.selected_array['frequency']
        timestamps = pitch.xs()
        segment_labels = self._classify_segments(
            raw_snd,
            timestamps,
            pitch_values,
        )

        pitch_values[pitch_values == 0] = np.nan
        formant_times, f1_values, f2_values, f3_values = self.extract_formants_for_track(
            timestamps,
            pitch_values,
            segment_labels,
        )
        return timestamps, pitch_values, segment_labels, formant_times, f1_values, f2_values, f3_values

    def extract_formants_for_track(self, timestamps, pitch_values, segment_labels=None):
        if self.audio_data is None or self.sr is None:
            return np.array([]), np.array([]), np.array([]), np.array([])

        timestamps = np.asarray(timestamps, dtype=float)
        pitch_values = np.asarray(pitch_values, dtype=float)
        if len(timestamps) == 0 or len(pitch_values) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        if segment_labels is not None and len(segment_labels) == len(pitch_values):
            segment_labels = np.asarray(segment_labels, dtype=int)
            voiced_mask = (segment_labels == SEGMENT_VOICED) & np.isfinite(pitch_values) & (pitch_values > 0)
        else:
            voiced_mask = np.isfinite(pitch_values) & (pitch_values > 0)
        if not np.any(voiced_mask):
            return np.array([]), np.array([]), np.array([]), np.array([])

        snd = parselmouth.Sound(self.audio_data, self.sr)
        max_formant = self._maximum_formant_for_file()
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
            out_times = []
            for t in voiced_times:
                f1 = formants.get_value_at_time(1, float(t))
                f2 = formants.get_value_at_time(2, float(t))
                f3 = formants.get_value_at_time(3, float(t))
                plausible = (
                    not np.isnan(f1)
                    and not np.isnan(f2)
                    and not np.isnan(f3)
                    and 50 < f1 < f2 < f3 < max_formant
                )
                if plausible:
                    out_times.append(float(t))
                    f1_vals.append(float(f1))
                    f2_vals.append(float(f2))
                    f3_vals.append(float(f3))
            return (
                np.asarray(out_times, dtype=float),
                np.asarray(f1_vals, dtype=float),
                np.asarray(f2_vals, dtype=float),
                np.asarray(f3_vals, dtype=float),
            )
        except Exception:
            return np.array([]), np.array([]), np.array([]), np.array([])

    def estimate_voiced_region(
        self,
        region_start,
        region_end,
        timestamps,
        pitch_floor,
        pitch_ceiling,
        time_step,
        voicing_threshold,
        silence_threshold,
        octave_cost,
        octave_jump_cost,
        voiced_unvoiced_cost,
    ):
        if self.audio_data is None or len(timestamps) == 0:
            return np.array([]), np.array([])

        region_mask = (timestamps >= region_start) & (timestamps <= region_end)
        region_times = np.asarray(timestamps[region_mask], dtype=float)
        if len(region_times) == 0:
            return np.array([]), np.array([])

        filtered_audio = self._apply_filtered_ac_lowpass(
            self.audio_data,
            self.sr,
            pitch_ceiling,
            self.filtered_ac_attenuation_at_top,
        )
        snd = parselmouth.Sound(filtered_audio, self.sr)
        part = snd.extract_part(
            from_time=max(0.0, float(region_start)),
            to_time=min(float(region_end), snd.duration),
            window_shape=parselmouth.WindowShape.RECTANGULAR,
            relative_width=1.0,
            preserve_times=True,
        )

        resolved_time_step = None if time_step <= 0 else time_step
        pitch = part.to_pitch_ac(
            time_step=resolved_time_step,
            pitch_floor=pitch_floor,
            pitch_ceiling=pitch_ceiling,
            voicing_threshold=max(0.01, voicing_threshold * 0.75),
            silence_threshold=min(1.0, silence_threshold + 0.02),
            octave_cost=octave_cost,
            octave_jump_cost=max(0.05, octave_jump_cost * 0.85),
            voiced_unvoiced_cost=max(0.01, voiced_unvoiced_cost * 0.75),
        )

        extracted_times = pitch.xs()
        extracted_values = pitch.selected_array["frequency"]
        extracted_values[extracted_values == 0] = np.nan

        estimated_values = np.full(len(region_times), np.nan, dtype=float)
        seed_freq = np.nanmedian(extracted_values)
        if np.isnan(seed_freq):
            seed_freq = (pitch_floor + pitch_ceiling) / 2.0

        for idx, region_time in enumerate(region_times):
            nearest = np.abs(extracted_times - region_time).argmin()
            candidate = extracted_values[nearest]
            if np.isnan(candidate):
                candidate = self.snap_to_peak(region_time, float(seed_freq), freq_window=120.0)
            estimated_values[idx] = candidate
            seed_freq = candidate

        return region_times, estimated_values

    @staticmethod
    def _apply_filtered_ac_lowpass(audio_data, sample_rate, pitch_top, attenuation_at_top):
        """
        Approximate Praat's filtered-autocorrelation prefilter in the frequency
        domain: H(f) = attenuation_at_top ** ((f / pitch_top) ** 2)
        """
        if pitch_top <= 0 or len(audio_data) == 0:
            return np.asarray(audio_data, dtype=float)

        samples = np.asarray(audio_data, dtype=float)
        spectrum = np.fft.rfft(samples)
        freqs = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)
        attenuation = np.power(
            float(attenuation_at_top),
            np.square(freqs / float(pitch_top)),
        )
        filtered = np.fft.irfft(spectrum * attenuation, n=len(samples))
        return np.asarray(filtered, dtype=float)

    def _detect_active_intervals(self, snd):
        """
        Match the main analysis path in AcousticAnalyses_Parselmouth.py:
        detect speech intervals on the original timeline first, then classify
        frames inside those intervals as voiced/unvoiced.
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
            for idx in range(1, num_intervals + 1):
                label = parselmouth.praat.call(tg, "Get label of interval", 1, idx)
                if label != "speech":
                    continue
                start = float(parselmouth.praat.call(tg, "Get start time of interval", 1, idx))
                end = float(parselmouth.praat.call(tg, "Get end time of interval", 1, idx))
                if end > start:
                    intervals.append((start, end))
            return intervals
        except Exception:
            return [(0.0, float(snd.duration))]

    def _classify_segments(self, snd, timestamps, pitch_values):
        if len(timestamps) == 0:
            return np.array([], dtype=int)

        labels = np.full(len(timestamps), SEGMENT_SILENCE, dtype=int)
        active_intervals = self._detect_active_intervals(snd)
        if not active_intervals:
            return labels

        active_mask = np.zeros(len(timestamps), dtype=bool)
        for start, end in active_intervals:
            active_mask |= (timestamps >= start) & (timestamps <= end)

        intensity = snd.to_intensity(time_step=self._safe_time_step(timestamps))
        intensity_values = np.array([intensity.get_value(float(t)) for t in timestamps], dtype=float)
        finite_active_intensity = intensity_values[active_mask & np.isfinite(intensity_values)]
        if len(finite_active_intensity) > 0:
            micro_silence_threshold = float(np.nanmax(finite_active_intensity) - 25.0)
            micro_silence_mask = active_mask & (
                ~np.isfinite(intensity_values) | (intensity_values < micro_silence_threshold)
            )
        else:
            micro_silence_mask = np.zeros(len(timestamps), dtype=bool)

        active_non_silence_mask = active_mask & ~micro_silence_mask
        voiced_mask = active_non_silence_mask & np.asarray(pitch_values > 0, dtype=bool)
        labels[active_non_silence_mask] = SEGMENT_VOICELESS
        labels[voiced_mask] = SEGMENT_VOICED
        return labels

    @staticmethod
    def _safe_time_step(timestamps):
        if len(timestamps) > 1:
            return max(float(np.median(np.diff(timestamps))), 1e-4)
        return 0.01
        
    def snap_to_peak(self, target_time, target_freq, freq_window=50.0):
        """
        Finds the local peak in the spectrogram nearest to the target_time and target_freq.
        """
        if self.S_db is None or self.spec_times is None or self.spec_freqs is None:
            return target_freq
            
        # Find time index
        t_idx = np.abs(self.spec_times - target_time).argmin()
        
        # Define frequency range
        f_min = max(0, target_freq - freq_window)
        f_max = target_freq + freq_window
        
        f_min_idx = np.abs(self.spec_freqs - f_min).argmin()
        f_max_idx = np.abs(self.spec_freqs - f_max).argmin()
        
        if f_max_idx <= f_min_idx:
            return target_freq
            
        # Get frame frequencies and amplitudes
        frame_amps = self.S_db[f_min_idx:f_max_idx, t_idx]
        frame_freqs = self.spec_freqs[f_min_idx:f_max_idx]
        
        # Find peaks
        peaks, _ = find_peaks(frame_amps)
        if len(peaks) == 0:
            # If no peaks, fallback to max value in range
            max_idx = np.argmax(frame_amps)
            return frame_freqs[max_idx]
            
        # Find peak closest to target_freq
        peak_freqs = frame_freqs[peaks]
        closest_peak_idx = np.abs(peak_freqs - target_freq).argmin()
        
        return peak_freqs[closest_peak_idx]
