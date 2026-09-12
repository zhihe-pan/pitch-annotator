import unittest,tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
from backend import acoustic_features as af
from backend import acoustic_analysis as aa
from main import Controller

PARAMS=Controller._praat_default_params(None)

class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.path=Path(cls.tmp.name)/'tone.wav'
        sr=16000;t=np.arange(sr)/sr;sf.write(cls.path,.2*np.sin(2*np.pi*200*t),sr)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def row(self,t,f,l):return af.compute_feature_row_with_pitch_overrides(str(self.path),PARAMS,np.array(t),np.array(f),np.array(l))
    def test_no_transition_across_unvoiced_gap(self):
        rise,fall=af._compute_segmented_rise_fall_for_track(np.array([.1,.2,.3,.4,.5]),np.array([20.,20.,np.nan,30.,30.]),[(0,1)],.25)
        self.assertEqual((rise,fall),(0.,0.))
    def test_legacy_no_transition_across_gap(self):
        self.assertEqual(aa.compute_segmented_rise_fall(np.array([.1,.2,.3,.4,.5]),np.array([20.,20.,np.nan,30.,30.]),[1],.25),(0.,0.))
    def test_known_f0_summary(self):
        f=np.array([110.,220.,440.,220.]);r=self.row([.1,.3,.5,.7],f,[2]*4)
        st=12*np.log2(f/27.5)
        for k,v in [('mean',np.mean(st)),('median',np.median(st)),('SD',np.std(st)),('P20',np.percentile(st,20)),('P80',np.percentile(st,80))]:self.assertAlmostEqual(r['F0_st_'+k],v,places=10)
    def test_duration_cannot_exceed_audio(self):
        t=np.arange(0,1.001,.01);r=self.row(t,np.full(len(t),200.),np.full(len(t),2));self.assertLessEqual(r['Voiced_duration_s'],1.)
    def test_short_track_still_has_complete_schema(self):
        p=Path(self.tmp.name)/'short.wav';sf.write(p,np.ones(320)*.01,16000)
        r=af.compute_feature_row_with_pitch_overrides(str(p),PARAMS,np.array([.005,.015]),np.array([200.,200.]),np.array([2,2]))
        self.assertIn('Int_mean',r);self.assertTrue(np.isnan(r['Int_mean']))
    def test_reject_invalid_time_order(self):
        with self.assertRaises(ValueError):self.row([.2,.1],[200.,200.],[2,2])
    def test_reject_out_of_range_time(self):
        with self.assertRaises(ValueError):self.row([.1,1.5],[200.,200.],[2,2])
    def test_reject_nonfinite_f0(self):
        with self.assertRaises(ValueError):self.row([.1,.2],[200.,np.inf],[2,2])
    def test_raw_short_audio_schema_matches_normal(self):
        p=Path(self.tmp.name)/'raw_short.wav';sf.write(p,np.ones(320)*.01,16000)
        short=af.extract_acoustic_feature_row(str(p),PARAMS)
        normal=af.extract_acoustic_feature_row(str(self.path),PARAMS)
        self.assertEqual(set(short),set(normal))
    def test_low_frequency_tone_has_negligible_high_frequency_energy(self):
        r=af._compute_activity_dependent_metrics(str(self.path),[(0.,1.)],[(0.,1.)],PARAMS)
        self.assertLess(r['HF500_ratio'],1e-4)
        self.assertLess(r['HF1000_ratio'],1e-4)
    def test_no_energy_above_nyquist(self):
        p=Path(self.tmp.name)/'low_rate.wav';t=np.arange(2000)/2000
        sf.write(p,.2*np.sin(2*np.pi*200*t),2000)
        r=af._compute_activity_dependent_metrics(str(p),[(0.,1.)],[(0.,1.)],PARAMS)
        self.assertEqual(r['HF1000_ratio'],0.)
    def test_zero_activity_schema(self):
        r=self.row([.1,.2],[np.nan,np.nan],[0,0]);self.assertEqual(r['Voiced_duration_s'],0);self.assertTrue(np.isnan(r['F0_st_mean']))
    def test_no_experimental_columns(self):
        r=self.row([.1,.2],[200.,200.],[2,2]);self.assertFalse(any('pitch_track' in k for k in r))

if __name__=='__main__':unittest.main()
