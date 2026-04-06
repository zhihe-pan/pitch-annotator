import csv
import numpy as np

def export_csv(filepath: str, timestamps: np.ndarray, pitch_values: np.ndarray):
    """
    Export pitch contour to CSV. Unvoiced frames will be exported with Frequency = 0.
    """
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time (s)', 'Frequency (Hz)'])
        for t, p in zip(timestamps, pitch_values):
            freq = 0 if np.isnan(p) else p
            writer.writerow([f"{t:.6f}", f"{freq:.6f}"])

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
