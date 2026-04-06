import csv
import sys
import types
from pathlib import Path


def _load_analysis_module():
    analysis_dir = Path(__file__).resolve().parents[1] / "Acoustic_analysis"
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
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return row
