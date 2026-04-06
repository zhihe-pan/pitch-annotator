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
        
        # Spectrogram data
        self.S_db = None
        self.spec_times = None
        self.spec_freqs = None
        self.window_length = 0.005
        self.maximum_frequency = 5000.0
        self.dynamic_range_db = 50.0
        self.filtered_ac_attenuation_at_top = 0.03

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
        spectrogram = snd.to_spectrogram(
            window_length=self.window_length,
            maximum_frequency=min(self.maximum_frequency, self.sr / 2.0),
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
            return np.array([]), np.array([]), np.array([], dtype=int)

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
            snd,
            timestamps,
            pitch_values,
            pitch_floor,
        )

        pitch_values[pitch_values == 0] = np.nan
        return timestamps, pitch_values, segment_labels

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

    def _classify_segments(self, snd, timestamps, pitch_values, pitch_floor):
        if len(timestamps) == 0:
            return np.array([], dtype=int)

        intensity = snd.to_intensity(
            minimum_pitch=max(50.0, float(pitch_floor)),
            time_step=self._safe_time_step(timestamps),
            subtract_mean=True,
        )
        intensity_values = np.array([intensity.get_value(t) for t in timestamps], dtype=float)
        finite_intensity = intensity_values[~np.isnan(intensity_values)]
        if len(finite_intensity) == 0:
            return np.full(len(timestamps), SEGMENT_SILENCE, dtype=int)

        max_intensity = float(np.nanmax(finite_intensity))
        silence_cutoff = max_intensity - 25.0
        labels = np.full(len(timestamps), SEGMENT_VOICELESS, dtype=int)

        silence_mask = np.isnan(intensity_values) | (intensity_values < silence_cutoff)
        voiced_mask = np.asarray(pitch_values > 0, dtype=bool)

        labels[silence_mask] = SEGMENT_SILENCE
        labels[~silence_mask & voiced_mask] = SEGMENT_VOICED
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
