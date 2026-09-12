"""Standalone cross-version Praat reference, run from repository root.
These comparisons establish implementation agreement, not scientific ground truth.
"""
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import parselmouth
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backend.acoustic_features import _compute_activity_dependent_metrics
OUT = ROOT / 'output_validation/accuracy-audit/reference'
PRAAT = '/Applications/Praat.app/Contents/MacOS/Praat'
PARAMS = {'pitch_floor': 75., 'pitch_ceiling': 500.}


def reference(path):
    result = subprocess.run([PRAAT, '--run', str(Path(__file__).with_name('reference_features.praat')), str(path), '75', '500'], capture_output=True, text=True, check=True)
    return {k: float(v) for k, v in next(csv.DictReader(io.StringIO(result.stdout))).items()}


def independent_spectrum(samples, rate):
    # Praat fast spectrum zero-pads to a power of 2; normalization cancels in COG.
    size = 1 << (len(samples) - 1).bit_length()
    spectrum = np.fft.rfft(samples, n=size)
    power = abs(spectrum) ** 2
    frequency = np.fft.rfftfreq(size, 1 / rate)
    weights = power.copy()
    weights[[0, -1]] *= .5
    cog = float(np.dot(frequency, weights) / weights.sum())
    selected = power > 0
    x = frequency[selected]
    y = 10 * np.log10(power[selected])
    # Explicit ordinary-least-squares covariance / variance, no np.polyfit.
    slope = float(np.sum((x-x.mean())*(y-y.mean())) / np.sum((x-x.mean())**2))
    ratios = {}
    for cutoff in (500, 1000):
        # Fraction of each frequency bin inside each band, including half endpoint bins.
        width = rate / size
        left = np.maximum(0, frequency-width/2)
        right = np.minimum(rate/2, frequency+width/2)
        low_width = np.maximum(0, np.minimum(right, cutoff)-left)
        high_width = np.maximum(0, right-np.maximum(left, cutoff))
        ratios[f'HF{cutoff}_ratio'] = float(np.dot(power,high_width)/np.dot(power,low_width))
    return {'COG_Hz': cog, 'Spectrum_slope': slope, **ratios}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rate = 16000
    t = np.arange(rate * 2) / rate
    rng = np.random.default_rng(8713)
    def tone(time, frequency=180, amplitude=.15, phase=0):
        return amplitude * sum(np.sin(2*np.pi*frequency*k*time+phase)/k for k in range(1, 8))
    cases = {'steady_harmonic': tone(t), 'modulated_harmonic': (1+.08*np.sin(2*np.pi*4*t))*tone(t+.0001*np.sin(2*np.pi*5*t)), 'noisy_harmonic': tone(t)+rng.normal(0,.003,t.size)}
    # Two stable tones of equal F0 but different phase/amplitude; neither has perturbation.
    cases['segment_a'] = tone(t[:8000])
    cases['segment_b'] = tone(t[:8000], amplitude=.20, phase=1.2)
    cases['disjoint_concatenated'] = np.concatenate([cases['segment_a'], cases['segment_b']])
    results = {}
    comparison = []
    for name, samples in cases.items():
        path = OUT / f'{name}.wav'
        wavfile.write(path, rate, samples.astype(np.float64))
        duration = len(samples) / rate
        backend = _compute_activity_dependent_metrics(path, [(0,duration)], [(0,duration)], PARAMS)
        ref = reference(path)
        _, embedded_text = parselmouth.praat.run_file(str(Path(__file__).with_name('reference_features.praat')), str(path), 75, 500, capture_output=True)
        embedded_ref = {k: float(v) for k, v in next(csv.DictReader(io.StringIO(embedded_text))).items()}
        for key, value in embedded_ref.items():
            assert np.isclose(backend[key], value, rtol=1e-9, atol=1e-10), (name, key, backend[key], value)
        independent = independent_spectrum(samples, rate)
        results[name] = {'backend': backend, 'praat_cli': ref, 'embedded_praat_script': embedded_ref, 'independent_fft': independent}
        for key, value in ref.items():
            comparison.append({'case':name,'metric':key,'backend':backend[key],'reference':value,'absolute_difference':abs(backend[key]-value)})
    # Verify actual backend disjoint selection yields exactly the concatenated waveform metrics.
    original = np.concatenate([cases['segment_a'], np.zeros(4000), cases['segment_b']])
    path = OUT / 'disjoint_original.wav'
    wavfile.write(path, rate, original)
    actual = _compute_activity_dependent_metrics(path, [(0,.5),(.75,1.25)], [(0,.5),(.75,1.25)], PARAMS)
    results['actual_disjoint_selection'] = actual
    for key in ['Jitter','Shimmer','HNR_dB']:
        assert np.isclose(actual[key],results['disjoint_concatenated']['backend'][key],rtol=1e-9,atol=1e-10)
    for case, data in results.items():
        if 'independent_fft' in data:
            assert abs(data['backend']['COG_Hz']-data['independent_fft']['COG_Hz']) < 1e-5
            assert abs(data['backend']['Spectrum_slope']-data['independent_fft']['Spectrum_slope']) < 1e-9
            for key in ('HF500_ratio','HF1000_ratio'):
                assert np.isclose(data['backend'][key],data['independent_fft'][key],rtol=1e-8,atol=1e-12), (case,key)
    summary = {'parselmouth_version': parselmouth.__version__, 'embedded_praat_version': parselmouth.PRAAT_VERSION, 'cli_version': subprocess.run([PRAAT,'--version'], capture_output=True,text=True).stdout.strip(), 'settings': {'pitch_floor':75,'pitch_ceiling':500,'jitter_period_bounds':[.0001,.02],'maximum_period_factor':1.3,'maximum_amplitude_factor':1.6,'harmonicity':[.01,75,.1,1],'intensity':[100,.01,True],'spectrum_fast':True}, 'scope':'Exact same whole waveform / concatenated effective selections. CLI is independent execution of Praat, not an independent algorithm or ground truth. Int_mean is arithmetic mean of positive dB frames, not Praat default energy mean.', 'results':results}
    (OUT/'reference_results.json').write_text(json.dumps(summary, indent=2, allow_nan=True))
    with (OUT/'reference_comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(comparison[0]));writer.writeheader();writer.writerows(comparison)
    print(json.dumps({'max_cross_version_difference_by_metric':{key:max(r['absolute_difference'] for r in comparison if r['metric']==key) for key in results['steady_harmonic']['praat_cli']}, 'jitter_shimmer_hnr':{case:{key:results[case]['praat_cli'][key] for key in ['Jitter','Shimmer','HNR_dB']} for case in ['segment_a','segment_b','disjoint_concatenated']}, 'output':str(OUT)}, indent=2))

if __name__ == '__main__':
    main()
