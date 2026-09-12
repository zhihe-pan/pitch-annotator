"""Deterministic audit fixtures; generated WAV files stay out of version control."""
from pathlib import Path
import hashlib,json
import numpy as np
import soundfile as sf
from scipy.signal import lfilter


def generate(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    manifest=[]
    def save(name,y,sr,truth):
        path=root/(name+'.wav');sf.write(path,y,sr,subtype='FLOAT')
        item=dict(id=name,path=str(path.resolve()),sample_rate=sr,duration=len(y)/sr,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**truth)
        manifest.append(item)
    for sr in (16000,44100,48000):
        t=np.arange(2*sr)/sr
        for f in (100,200,400):
            y=.25*np.sin(2*np.pi*f*t)+.05*np.sin(4*np.pi*f*t)
            save(f'tone_{f}_{sr}',y,sr,dict(kind='steady',f0=f))
            if sr==44100:
                for mode in ('padded','gap','short','half','double','stereo'):
                    z=y.copy()
                    if mode=='padded':z=np.pad(z,(sr//2,sr//2))
                    if mode=='gap':z[sr*3//4:sr*5//4]=0
                    if mode=='short':z=z[:sr//50]
                    if mode=='half':z*=.5
                    if mode=='double':z*=2
                    if mode=='stereo':z=np.column_stack([z,z*.5])
                    save(f'tone_{f}_{mode}',z,sr,dict(kind=mode,f0=f))
    sr=44100;t=np.arange(sr*2)/sr
    for name,f0 in [('rise',100+100*t),('fall',300-100*t),('octave',np.where(t<1,150.,300.))]:
        y=.25*np.sin(2*np.pi*np.cumsum(f0)/sr)
        save(name,y,sr,dict(kind=name))
    save('silence',np.zeros(2*sr),sr,dict(kind='silence'))
    save('noise',np.random.default_rng(20260912).normal(0,.04,2*sr),sr,dict(kind='noise'))
    # Known period and amplitude modulation. Their discrete pulse sequence is the truth;
    # the pitch/voice estimator need not recover it exactly.
    for name,jit,shim in [('vowel',0,0),('jitter',.01,0),('shimmer',0,.1),('perturbed',.01,.1)]:
        positions=[0];periods=[];amps=[]
        while positions[-1]<2*sr:
            n=len(periods); period=round(sr/150*(1+jit*(-1)**n))
            periods.append(period/sr);amps.append(1+shim*(-1)**n);positions.append(positions[-1]+period)
        y=np.zeros(2*sr)
        for pos,amp in zip(positions,amps):
            if pos<len(y):y[pos]=amp
        for freq,bw in [(500,60),(1500,90),(2500,150)]:
            r=np.exp(-np.pi*bw/sr);y=lfilter([1],[1,-2*r*np.cos(2*np.pi*freq/sr),r*r],y)
        y=.3*y/np.max(np.abs(y));pp=np.array(periods);aa=np.array(amps)
        save(name,y,sr,dict(kind='source_filter',f0=150,formants=[500,1500,2500],bandwidths=[60,90,150],jitter_truth=float(np.mean(abs(np.diff(pp)))/np.mean(pp)),shimmer_truth=float(np.mean(abs(np.diff(aa)))/np.mean(aa))))
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest

if __name__=='__main__':
    import sys
    print(len(generate(sys.argv[1])), 'fixtures written')
