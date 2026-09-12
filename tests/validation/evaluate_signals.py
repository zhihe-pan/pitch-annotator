"""Report estimator error, not a claim that default parameters fit all signals."""
import csv,hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import parselmouth
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from backend.audio_core import AudioProcessor
from backend.acoustic_features import compute_feature_row_with_pitch_overrides
OUT=ROOT/'output_validation/accuracy-audit/signal-evaluation'
PARAMS=dict(pitch_floor=50.,pitch_ceiling=800.,time_step=.01,filtered_ac_attenuation_at_top=.03,voicing_threshold=.5,silence_threshold=.09,octave_cost=.055,octave_jump_cost=.35,voiced_unvoiced_cost=.14)

def truth(item,times):
    kind=item['kind']; mask=np.ones(len(times),bool)
    if kind in ('silence','noise','real'):return None,mask
    if kind=='rise':f=100+100*times
    elif kind=='fall':f=300-100*times
    elif kind=='octave':f=np.where(times<1,150.,300.)
    else:f=np.full(len(times),item['f0'])
    if kind=='padded':mask=(times>=.5)&(times<2.5)
    if kind=='gap':mask=(times<.75)|(times>=1.25)
    return f,mask

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'output_validation/accuracy-audit/signals/manifest.json').read_text())
    for p in sorted((ROOT/'stim').glob('*.wav'))[:2]:manifest.append(dict(id=p.stem,path=str(p),kind='real',sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'backend').glob('*.py')}
    rows=[];start=time.time()
    for item in manifest:
      for mode in ('internal','external'):
        record=dict(id=item['id'],kind=item['kind'],requested_mode=mode,audio_sha256=item['sha256'])
        try:
          proc=AudioProcessor()
          if mode=='internal':proc._praat_checked=True;proc._praat_executable=None
          else:proc._praat_checked=True;proc._praat_executable='/Applications/Praat.app/Contents/MacOS/Praat'
          proc.load_audio(item['path'])
          ts,f0,labels,*rest=proc.extract_pitch(**PARAMS)
          record['actual_source']=rest[-1]
          record['mode_matched']=('External' in rest[-1])==(mode=='external')
          valid=np.isfinite(f0)&(f0>0);expected,voiced=truth(item,ts)
          record.update(frames=len(ts),voiced_frames=int(valid.sum()),voiced_fraction=float(valid.mean()) if len(ts) else None)
          if expected is not None:
            selected=valid&voiced
            cents=1200*np.log2(f0[selected]/expected[selected])
            record['f0_evaluation']=dict(reference='nominal 150 Hz for perturbed source-filter; instantaneous generator F0 otherwise',evaluated_frames=int(selected.sum()),missing_voiced_fraction=float(np.mean(~valid[voiced])) if voiced.any() else None,median_absolute_cents=float(np.median(abs(cents))) if cents.size else None,p95_absolute_cents=float(np.percentile(abs(cents),95)) if cents.size else None,gross_error_fraction_over_20_percent=float(np.mean(abs(f0[selected]/expected[selected]-1)>.2)) if cents.size else None,false_voiced_silence_fraction=float(np.mean(valid[~voiced])) if (~voiced).any() else None)
          try:
            features=compute_feature_row_with_pitch_overrides(item['path'],PARAMS,ts,f0,labels)
            record['features']=features
            record['screen_formant_means']={f'F{i+1}_mean':float(np.mean(values)) if len(values) else float('nan') for i,values in enumerate(rest[1:4])}
            record['screen_minus_export_formants']={k:v-features.get(k,float('nan')) for k,v in record['screen_formant_means'].items()}
            if item['kind']=='source_filter':
              alias=OUT / ('gender2_'+Path(item['path']).name)
              if not alias.exists(): alias.symlink_to(item['path'])
              alias_features=compute_feature_row_with_pitch_overrides(str(alias),PARAMS,ts,f0,labels)
              record['gender2_alias_export_formants']={k:alias_features[k] for k in ('F1_mean','F2_mean','F3_mean','F1_BW_mean','F2_BW_mean','F3_BW_mean')}
              record['gender2_minus_original_formants']={k:v-features[k] for k,v in record['gender2_alias_export_formants'].items()}
            record['nonfinite_feature_names']=[k for k,v in features.items() if isinstance(v,(float,np.floating)) and not np.isfinite(v)]
            record['infinite_feature_names']=[k for k,v in features.items() if isinstance(v,(float,np.floating)) and np.isinf(v)]
            if item['kind']=='source_filter':
              record['source_filter_errors']={**{f'F{i+1}_Hz':features.get(f'F{i+1}_mean',np.nan)-v for i,v in enumerate(item['formants'])},**{f'F{i+1}_BW_Hz':features.get(f'F{i+1}_BW_mean',np.nan)-v for i,v in enumerate(item['bandwidths'])},'jitter_difference':features['Jitter']-item['jitter_truth'],'shimmer_difference':features['Shimmer']-item['shimmer_truth'],'jitter_generator_truth':item['jitter_truth'],'shimmer_generator_truth':item['shimmer_truth']}
          except Exception as e:record['feature_error']=repr(e)
        except Exception as e:record['load_or_pitch_error']=repr(e)
        rows.append(record)
        print(item['id'],mode,record.get('actual_source',record.get('load_or_pitch_error')),flush=True)
    now_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'backend').glob('*.py')}
    report=dict(parameters=PARAMS,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),parselmouth=parselmouth.__version__,embedded_praat=parselmouth.PRAAT_VERSION,external_praat=subprocess.check_output(['/Applications/Praat.app/Contents/MacOS/Praat','--version'],text=True).strip(),source_hashes_start=source_hashes,source_hashes_end=now_hashes,source_changed_during_run=source_hashes!=now_hashes,elapsed_seconds=time.time()-start,notes=['No scientific acceptance thresholds asserted. Gross error >20% is descriptive only.','F0 errors conditioned on detected voiced frames; missing voiced fraction is separate.','Source filter pole parameters and generator perturbations are not guaranteed recoverable waveform ground truth, especially Shimmer after filtering.','Real audio has no ground truth. NaN alone is not necessarily a defect; inspect metric applicability.','Pitch extraction uses AudioProcessor; feature export uses extracted labels and pitch.'],results=rows)
    (OUT/'evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=True,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
    flat=[dict(id=r['id'],kind=r['kind'],mode=r['requested_mode'],source=r.get('actual_source'),error=r.get('load_or_pitch_error',r.get('feature_error','')),**r.get('f0_evaluation',{})) for r in rows]
    keys=list(dict.fromkeys(k for r in flat for k in r))
    with (OUT/'f0_errors.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(flat)
    print('saved',OUT,flush=True)
if __name__=='__main__':run()
