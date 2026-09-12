import csv,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import parselmouth
from main import Controller
from backend.audio_core import AudioProcessor
from core.exporter import export_csv
from core.state import PitchState

class ImportTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'track.csv';self.reader=Controller.__new__(Controller)
    def tearDown(self):self.tmp.cleanup()
    def read(self,rows):
        with self.path.open('w') as f:
            w=csv.writer(f);w.writerow(['Time (s)','Frequency (Hz)','SegmentLabel']);w.writerows(rows)
        return self.reader._read_pitch_csv(self.path)
    def test_reject_invalid_parameter_text(self):
        self.path.write_text('Time (s),Frequency (Hz),pitch_floor\n0.1,200,not-a-number\n')
        with self.assertRaises(ValueError):self.reader._read_pitch_csv(self.path)
    def test_reject_nan_time(self):
        with self.assertRaises(ValueError):self.read([['nan',200,2]])
    def test_reject_negative_time(self):
        with self.assertRaises(ValueError):self.read([[-.1,200,2]])
    def test_reject_duplicate_time(self):
        with self.assertRaises(ValueError):self.read([[.1,200,2],[.1,210,2]])
    def test_reject_irregular_time(self):
        with self.assertRaises(ValueError):self.read([[.1,200,2],[.2,210,2],[.5,220,2]])
    def test_reject_invalid_label(self):
        with self.assertRaises(ValueError):self.read([[.1,200,9]])
    def test_reject_infinite_frequency(self):
        with self.assertRaises(ValueError):self.read([[.1,'inf',2]])
    def test_unique_audio_match(self):
        self.assertEqual(Controller._pick_audio_candidate([Path('/a/x.wav')],''),Path('/a/x.wav'))
    def test_reject_ambiguous_audio(self):
        with self.assertRaises(ValueError):Controller._pick_audio_candidate([Path('/a/x.wav'),Path('/b/x.wav')],'')
    def test_zero_pitch_defaults_to_unvoiced(self):
        r=self.read([[.1,0,''],[.2,200,'']]);np.testing.assert_array_equal(r['segment_labels'],[1,2])
    def test_roundtrip(self):
        t=np.arange(.01,.11,.01);f=np.full(len(t),200.);l=np.full(len(t),2);l[3]=1;f[3]=np.nan
        export_csv(self.path,t,f,Controller._praat_default_params(None),'/a/example.wav',l)
        r=self.reader._read_pitch_csv(self.path)
        np.testing.assert_allclose(r['timestamps'],t,atol=5e-7);np.testing.assert_allclose(r['pitch_values'],f,equal_nan=True);np.testing.assert_array_equal(r['segment_labels'],l)
    def test_labels_override_positive_unvoiced_f0(self):
        r=self.read([[.1,200,1],[.2,200,2]]);self.assertTrue(np.isnan(r['pitch_values'][0]))

class EditingTests(unittest.TestCase):
    def test_state_undo(self):
        s=PitchState();t=np.arange(.01,1,.01);f=np.full(len(t),200.);s.update_pitch_data(t,f,np.full(len(t),2));before=s.snapshot_edit_state()
        s.set_silence(.2,.4);s.set_unvoiced(.6,.7);s.add_or_update_point(.8,400);s.restore_edit_state(before)
        np.testing.assert_array_equal(s.pitch_values,before['pitch_values']);np.testing.assert_array_equal(s.segment_labels,before['segment_labels'])
    def test_fallback_preserves_absolute_time(self):
        proc=AudioProcessor();sr=16000;t=np.arange(sr)/sr;s=parselmouth.Sound(.2*np.sin(2*np.pi*200*t),sr)
        s.shift_times_by(.7)
        with patch('backend.audio_core.parselmouth.praat.call', side_effect=parselmouth.PraatError('force legacy fallback')):
            p=proc._to_pitch_filtered_ac(s,75,500,.01,.03,.5,.09,.055,.35,.14)
        self.assertGreaterEqual(p.xs()[0],.7)

if __name__=='__main__':unittest.main()
