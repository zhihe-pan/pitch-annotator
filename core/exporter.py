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

def export_praat_pitch(filepath: str, timestamps: np.ndarray, pitch_values: np.ndarray):
    """
    Export to standard Praat Pitch Short Text File.
    Format:
    File type = "ooTextFile"
    Object class = "Pitch 1"
    
    xmin
    xmax
    nx
    dx
    x1
    <nx times>:
        intensity
        nCandidates
        candidate1_frequency
        candidate1_strength
    """
    if len(timestamps) == 0:
        return
        
    n_frames = len(timestamps)
    # Estimate time step dx from the first two timestamps
    if n_frames > 1:
        dx = timestamps[1] - timestamps[0]
    else:
        dx = 0.01 # Default fallback
    
    x1 = timestamps[0]
    xmin = x1 - dx / 2
    xmax = timestamps[-1] + dx / 2
    
    lines = [
        'File type = "ooTextFile"',
        'Object class = "Pitch 1"',
        '',
        f'{xmin:.6f}',
        f'{xmax:.6f}',
        f'{n_frames}',
        f'{dx:.6f}',
        f'{x1:.6f}'
    ]
    
    for p in pitch_values:
        # Intensity = 1 for simplicity
        lines.append('1.0')
        lines.append('1')  # 1 candidate
        if np.isnan(p):
            lines.append('0.0') # Unvoiced freq = 0
            lines.append('0.0') # Strength = 0
        else:
            lines.append(f'{p:.6f}')
            lines.append('1.0') # Strength = 1
            
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')
