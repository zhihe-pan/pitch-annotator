import csv
import numpy as np

PARAMETER_COLUMNS = [
    "pitch_floor",
    "pitch_ceiling",
    "time_step",
    "filtered_ac_attenuation_at_top",
    "voicing_threshold",
    "silence_threshold",
    "octave_cost",
    "octave_jump_cost",
    "voiced_unvoiced_cost",
]


def export_csv(filepath: str, timestamps: np.ndarray, pitch_values: np.ndarray, pitch_params=None, audio_path=None, segment_labels=None):
    """
    Export pitch contour to CSV. Unvoiced frames will be exported with Frequency = 0.
    """
    timestamps = np.asarray(timestamps, dtype=float)
    pitch_values = np.asarray(pitch_values, dtype=float)
    if segment_labels is None:
        segment_labels = np.full(len(pitch_values), -1, dtype=int)
    else:
        segment_labels = np.asarray(segment_labels, dtype=int)
        if len(segment_labels) != len(pitch_values):
            segment_labels = np.full(len(pitch_values), -1, dtype=int)

    pitch_params = {} if pitch_params is None else dict(pitch_params)
    metadata = {name: pitch_params.get(name, "") for name in PARAMETER_COLUMNS}
    metadata["audio_file"] = "" if audio_path is None else str(audio_path)

    fieldnames = ["audio_file"] + PARAMETER_COLUMNS + ["Time (s)", "Frequency (Hz)", "SegmentLabel"]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        freqs = np.where(np.isnan(pitch_values), 0.0, pitch_values)
        rows = []
        for time_value, freq_value, segment_label in zip(timestamps, freqs, segment_labels):
            row = dict(metadata)
            row["Time (s)"] = f"{float(time_value):.6f}"
            row["Frequency (Hz)"] = f"{float(freq_value):.6f}"
            row["SegmentLabel"] = int(segment_label)
            rows.append(row)
        writer.writerows(rows)

def export_praat_pitch(
    filepath: str,
    timestamps: np.ndarray,
    pitch_values: np.ndarray,
    pitch_ceiling: float | None = None,
):
    """
    Export to a Praat-compatible text Pitch object.

    We intentionally use the verbose text format instead of an abbreviated
    short-text variant, because Praat accepts it reliably and it is less
    error-prone when generated outside Praat.
    """
    timestamps = np.asarray(timestamps, dtype=float)
    pitch_values = np.asarray(pitch_values, dtype=float)
    n_frames = min(len(timestamps), len(pitch_values))
    if n_frames == 0:
        return

    timestamps = timestamps[:n_frames]
    pitch_values = pitch_values[:n_frames]

    # Estimate time step dx from the first two timestamps.
    if n_frames > 1:
        dx = float(timestamps[1] - timestamps[0])
    else:
        dx = 0.01  # Default fallback.

    x1 = float(timestamps[0])
    xmin = x1 - dx / 2.0
    xmax = float(timestamps[-1] + dx / 2.0)
    if pitch_ceiling is None or not np.isfinite(pitch_ceiling) or float(pitch_ceiling) <= 0:
        finite_pitch = pitch_values[np.isfinite(pitch_values) & (pitch_values > 0)]
        pitch_ceiling = float(np.nanmax(finite_pitch)) if len(finite_pitch) else 800.0

    def format_number(value: float) -> str:
        return np.format_float_positional(float(value), trim="-", precision=10)

    lines = [
        'File type = "ooTextFile"',
        'Object class = "Pitch 1"',
        '',
        f"xmin = {format_number(xmin)}",
        f"xmax = {format_number(xmax)}",
        f"nx = {n_frames}",
        f"dx = {format_number(dx)}",
        f"x1 = {format_number(x1)}",
        f"ceiling = {format_number(float(pitch_ceiling))}",
        "maxnCandidates = 1",
        "frame []:",
    ]

    for index, value in enumerate(pitch_values, start=1):
        lines.append(f"    frame [{index}]:")
        lines.append("        intensity = 1")
        lines.append("        nCandidates = 1")
        lines.append("        candidate []:")
        lines.append("            candidate [1]:")
        if np.isnan(value) or value <= 0:
            lines.append("                frequency = 0")
            lines.append("                strength = 0")
        else:
            lines.append(f"                frequency = {format_number(float(value))}")
            lines.append("                strength = 1")

    with open(filepath, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
