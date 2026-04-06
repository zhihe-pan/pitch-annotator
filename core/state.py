import numpy as np

class PitchState:
    def __init__(self):
        self.audio_path = None
        self.audio_data = None
        self.sample_rate = None
        self.timestamps = np.array([])
        self.pitch_values = np.array([])  # np.nan for unvoiced
        self.segment_labels = np.array([], dtype=int)
        self.voice_percent = 0.0
        
        self.pitch_floor = 50.0
        self.pitch_ceiling = 800.0
        self.time_step = 0.0
        self.voicing_threshold = 0.50
        self.silence_threshold = 0.09
        self.octave_cost = 0.055
        self.octave_jump_cost = 0.35
        self.voiced_unvoiced_cost = 0.14
        
        self._callbacks = []

    def register_callback(self, callback):
        self._callbacks.append(callback)

    def trigger_update(self):
        for cb in self._callbacks:
            cb()

    def set_audio_data(self, audio_data, sample_rate):
        self.audio_data = audio_data
        self.sample_rate = sample_rate

    def snapshot_edit_state(self):
        return {
            "pitch_values": np.array(self.pitch_values, copy=True),
            "segment_labels": np.array(self.segment_labels, copy=True),
            "voice_percent": float(self.voice_percent),
        }

    def restore_edit_state(self, snapshot):
        self.pitch_values = np.array(snapshot["pitch_values"], copy=True)
        self.segment_labels = np.array(snapshot["segment_labels"], copy=True)
        self.voice_percent = float(snapshot["voice_percent"])
        self.trigger_update()

    def update_pitch_data(self, timestamps, pitch_values, segment_labels=None):
        self.timestamps = timestamps
        self.pitch_values = pitch_values
        if segment_labels is None:
            segment_labels = np.zeros(len(timestamps), dtype=int)
        self.segment_labels = np.asarray(segment_labels, dtype=int)
        self.voice_percent = self._compute_voice_percent()
        self.trigger_update()

    def set_voiced(self, start_time, end_time, new_values):
        """
        Set a region to voiced with new pitch values
        new_values should have the same length as the mask True count
        """
        mask = self._region_mask(start_time, end_time)
        target_count = int(np.sum(mask))
        if target_count == 0:
            return
        values = np.asarray(new_values, dtype=float)
        if len(values) != target_count:
            if len(values) == 0:
                return
            source_x = np.linspace(0.0, 1.0, num=len(values))
            target_x = np.linspace(0.0, 1.0, num=target_count)
            values = np.interp(target_x, source_x, values)
        self.pitch_values[mask] = values
        if len(self.segment_labels) == len(self.pitch_values):
            self.segment_labels[mask] = 2
        self.voice_percent = self._compute_voice_percent()
        self.trigger_update()

    def set_unvoiced(self, start_time, end_time):
        """
        Set a region to unvoiced (np.nan)
        """
        mask = self._region_mask(start_time, end_time)
        self.pitch_values[mask] = np.nan
        if len(self.segment_labels) == len(self.pitch_values):
            self.segment_labels[mask] = 1
        self.voice_percent = self._compute_voice_percent()
        self.trigger_update()

    def set_silence(self, start_time, end_time):
        """
        Set a region to silence.
        Silence frames are excluded from Voice% denominator.
        """
        mask = self._region_mask(start_time, end_time)
        self.pitch_values[mask] = np.nan
        if len(self.segment_labels) == len(self.pitch_values):
            self.segment_labels[mask] = 0
        self.voice_percent = self._compute_voice_percent()
        self.trigger_update()

    def add_or_update_point(self, time_val, freq_val):
        """
        Modify the closest available point to time_val
        """
        if len(self.timestamps) == 0:
            return
        idx = np.abs(self.timestamps - time_val).argmin()
        self.pitch_values[idx] = freq_val
        if len(self.segment_labels) == len(self.pitch_values):
            self.segment_labels[idx] = 2
        self.voice_percent = self._compute_voice_percent()
        self.trigger_update()

    def remove_point(self, time_val):
        """
        Set the closest accessible point as unvoiced
        """
        if len(self.timestamps) == 0:
            return
        idx = np.abs(self.timestamps - time_val).argmin()
        self.pitch_values[idx] = np.nan
        if len(self.segment_labels) == len(self.pitch_values):
            self.segment_labels[idx] = 1
        self.voice_percent = self._compute_voice_percent()
        self.trigger_update()

    def get_quantiles(self):
        """
        Calculate F0 20%, 50%, 80% quantiles for status bar.
        Ignores unvoiced (NaN) frames.
        """
        valid_pitches = self.pitch_values[~np.isnan(self.pitch_values)]
        if len(valid_pitches) == 0:
            return 0.0, 0.0, 0.0
        return (
            np.percentile(valid_pitches, 20),
            np.percentile(valid_pitches, 50),
            np.percentile(valid_pitches, 80)
        )

    def _region_mask(self, start_time, end_time):
        if len(self.timestamps) == 0:
            return np.zeros(0, dtype=bool)
        region_min = min(start_time, end_time)
        region_max = max(start_time, end_time)
        mask = (self.timestamps >= region_min) & (self.timestamps <= region_max)
        if np.any(mask):
            return mask
        center = (region_min + region_max) / 2.0
        nearest_idx = int(np.abs(self.timestamps - center).argmin())
        fallback_mask = np.zeros(len(self.timestamps), dtype=bool)
        fallback_mask[nearest_idx] = True
        return fallback_mask

    def _compute_voice_percent(self):
        if len(self.pitch_values) == 0:
            return 0.0

        if len(self.segment_labels) == len(self.pitch_values):
            active_mask = self.segment_labels != 0
        else:
            active_mask = np.ones(len(self.pitch_values), dtype=bool)

        active_count = int(np.sum(active_mask))
        if active_count == 0:
            return 0.0

        voiced_mask = active_mask & (~np.isnan(self.pitch_values)) & (self.pitch_values > 0)
        return float(np.sum(voiced_mask) / active_count * 100.0)
