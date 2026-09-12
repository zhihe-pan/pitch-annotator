"""Independent FFT and standalone embedded-Praat agreement on mono harmonic audio."""
import csv,io,tempfile,unittest
from pathlib import Path
import numpy as np
import parselmouth
from scipy.io import wavfile
from reference_checks import independent_spectrum,PARAMS
from backend.acoustic_features import _compute_activity_dependent_metrics

class ReferenceAgreement(unittest.TestCase):
    def test_same_version_script_and_independent_fft(self):
        rate=16000;t=np.arange(rate)/rate
        y=.1*np.sin(2*np.pi*180*t)+.03*np.sin(2*np.pi*360*t)+np.random.default_rng(8713).normal(0,.001,len(t))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'reference.wav';wavfile.write(path,rate,y)
            values=_compute_activity_dependent_metrics(path,[(0,1)],[(0,1)],PARAMS)
            _,output=parselmouth.praat.run_file(str(Path(__file__).with_name('reference_features.praat')),str(path),75,500,capture_output=True)
            for key,value in next(csv.DictReader(io.StringIO(output))).items():
                with self.subTest(metric=key):self.assertAlmostEqual(values[key],float(value),places=9)
            independent=independent_spectrum(y,rate)
            self.assertAlmostEqual(values['COG_Hz'],independent['COG_Hz'],places=5)
            self.assertAlmostEqual(values['Spectrum_slope'],independent['Spectrum_slope'],places=9)
            for key in ('HF500_ratio','HF1000_ratio'):
                with self.subTest(independent_band_ratio=key):self.assertAlmostEqual(values[key],independent[key],places=9)

if __name__=='__main__':unittest.main()
