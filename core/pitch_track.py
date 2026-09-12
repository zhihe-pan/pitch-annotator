"""Validation shared by imported and exported frame-based pitch tracks."""
import numpy as np


def validate_pitch_track(timestamps, pitch_values, segment_labels=None, duration=None):
    times = np.asarray(timestamps, dtype=float)
    values = np.asarray(pitch_values, dtype=float).copy()
    if times.ndim != 1 or values.ndim != 1 or len(times) != len(values):
        raise ValueError('Pitch timestamps and frequencies must be equally sized one-dimensional arrays.')
    if not np.all(np.isfinite(times)) or np.any(times < 0):
        raise ValueError('Pitch timestamps must be finite and non-negative.')
    if len(times) > 1:
        steps = np.diff(times)
        if np.any(steps <= 0):
            raise ValueError('Pitch timestamps must be strictly increasing, without duplicates.')
        # CSV output rounds to six decimal places. Permit that rounding only.
        if not np.allclose(steps, np.median(steps), rtol=1e-5, atol=1.01e-6):
            raise ValueError('Pitch timestamps must use a uniform frame interval.')
    if duration is not None and np.any(times > float(duration) + 5e-7):
        raise ValueError('Pitch timestamps extend beyond the original audio duration.')
    if np.any(np.isinf(values)):
        raise ValueError('Pitch frequencies must be finite or NaN, not infinity.')
    values[values <= 0] = np.nan
    if segment_labels is None:
        labels = np.where(np.isfinite(values), 2, 1)
    else:
        raw = np.asarray(segment_labels)
        if raw.ndim != 1 or len(raw) != len(times) or not np.all(np.isin(raw, [0, 1, 2])):
            raise ValueError('SegmentLabel must contain only 0, 1 or 2, one label per frame.')
        labels = raw.astype(int, copy=True)
        # Labels determine whether a point belongs to the voiced track.
        values[labels != 2] = np.nan
        labels[(labels == 2) & ~np.isfinite(values)] = 1
    return times, values, labels
