# Pitch Annotator

A desktop GUI tool for manual correction of pitch (F0) tracks in speech and voice research. Built with PySide6, pyqtgraph, and Parselmouth.

<p align="center">
  <img src="pitch_annotator.png" alt="Pitch Annotator screenshot" width="800"/>
</p>

## Features

- **Batch import** audio files (.wav, .mp3, .m4a, .flac, .aiff, .ogg) and review them one by one
- **Spectrogram + pitch + formants** visualization in a single view
- **Praat-style** pitch extraction with configurable parameters (pitch floor/ceiling, voicing threshold, octave cost, etc.)
- **Manual editing** — add, remove, and drag individual pitch points; shift entire regions up/down
- **Segment labeling** — mark regions as Silence / Voiceless / Voiced with color bands
- **Zoom & navigate** — mouse wheel zoom anchored to the pitch contour; horizontal scrollbar for precise time-axis positioning
- **Resizable panels** — drag splitters to adjust sidebar and control panel widths
- **Playback** — play original audio selection or synthesized F0 tone
- **Undo** support (up to 100 steps)
- **Export** in multiple formats:
  - CSV (pitch values with segment labels & parameters)
  - Praat `.Pitch` file
  - Spectrogram plot (PNG)
  - Acoustic features CSV (intensity, jitter, shimmer, HNR, COG, spectral slope, F0 statistics, formant means)
  - Batch export for all loaded audio files
- **External Praat** — optionally use a local Praat installation for pitch extraction

## Installation

**Requirements:** Python 3.11 or 3.12

```bash
git clone https://github.com/zhihe-pan/pitch-annotator.git
cd pitch-annotator
python -m pip install -r requirements.txt
python main.py
```

### Optional: External Praat

Install [Praat](https://www.fon.hum.uva.nl/praat/) and set the environment variable `PRAAT_PATH` pointing to `Praat.exe` (Windows) or `praatcon` (macOS/Linux). The status bar will show "Pitch source: External Praat filtered AC" when detected.

## Usage

### Mouse

| Action | Gesture |
|--------|---------|
| Select pitch point | Left-click |
| Create pitch point | Alt + Left-click |
| Delete pitch point | Alt + Shift + Left-click |
| Drag pitch point | Left-click + drag on a point |
| Create selection region | Left-click + drag on empty space |
| Shift selected region vertically | Shift + Left-click + drag |
| Zoom in/out | Mouse wheel (anchored to pitch contour) |
| Scroll time axis | Drag horizontal scrollbar below spectrogram |

### Keyboard

| Shortcut | Action |
|----------|--------|
| Space | Play selected region (audio) |
| Shift + Space | Play selected region (F0 tone) |
| Ctrl+Z | Undo |
| Up / Down | Previous / next audio file |
| Ctrl+Shift+E | Export all for current file |
| Ctrl+Alt+Shift+E | Export all for batch |

### Workflow

1. **File → Import Audio Files…** — select one or more audio files, set initial pitch parameters
2. Inspect the spectrogram and pitch contour — zoom, pan, scroll as needed
3. Toggle the **Region** tool in the control panel to select a time range
4. Mark the region as Voiced / Voiceless / Silence
5. Manually adjust pitch points: Alt+Click to add, Alt+Shift+Click to remove, drag to reposition
6. **Export** — use File menu or shortcuts to save results

### Pitch Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| Pitch floor | Minimum F0 (Hz) | 50 |
| Pitch ceiling | Maximum F0 (Hz) | 800 |
| Time step | Frame step (0 = auto) | 0.0 s |
| Filtered AC attenuation | Attenuation at ceiling frequency | 0.03 |
| Voicing threshold | Amplitude threshold for voicing | 0.50 |
| Silence threshold | Amplitude below which = silence | 0.09 |
| Octave cost | Penalty for octave jumps | 0.055 |
| Octave jump cost | Penalty for octave discontinuities | 0.35 |
| Voiced/unvoiced cost | Cost of voicing state change | 0.14 |

## Build standalone app

```bash
python -m pip install -r requirements.txt
python build.py
```

Outputs appear in `dist/` (PyInstaller) and `release/` (distributable archive).

## Tech stack

- **Python 3.11+**
- **PySide6** — Qt GUI framework
- **pyqtgraph** — high-performance plotting
- **praat-parselmouth** — acoustic analysis (spectrogram, pitch, formants)
- **librosa, scipy, numpy, soundfile** — audio processing
- **PyInstaller** — desktop packaging

## License

MIT
