"""Automated offscreen Qt event/worker acceptance, not manual GUI acceptance."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import sys,json,csv,time,traceback,faulthandler
faulthandler.dump_traceback_later(30, repeat=True)
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import soundfile as sf
# Resolve librosa's lazy audio/Numba imports before Qt installs its import hook.
# This changes dependency warm-up only; Controller still loads audio in QThread.
import librosa
_librosa_loader = librosa.load
from PySide6.QtWidgets import QApplication,QPushButton,QMessageBox
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from main import Controller,MainWindow,PitchState,AudioProcessor
from core.exporter import export_csv


def run():
    out=ROOT/'output_validation/accuracy-audit/gui';out.mkdir(parents=True,exist_ok=True)
    result={'dependency_warmup':'librosa.load resolved before Qt import; no audio computation mocked','mode':'automated offscreen Qt; not manual full GUI acceptance','checks':[],'errors':[]}
    app=QApplication.instance() or QApplication([])
    window=MainWindow();state=PitchState();controller=Controller(window,state,AudioProcessor())
    window.resize(1400,900);window.show()
    formant_events=[]
    controller.worker.finished_formants.connect(lambda *args:formant_events.append(len(args[2])))
    # Only dialogs are substituted: real QAction/button signals, Controller and QThreads run.
    def dialog_error(*args,**kwargs):
        result['errors'].append(str(args[2] if len(args)>2 else args));return QMessageBox.Ok
    QMessageBox.critical=dialog_error;QMessageBox.warning=dialog_error
    def wait(predicate,timeout=60):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            app.processEvents();QTest.qWait(20)
            if result['errors']:raise AssertionError(result['errors'])
            if predicate():return
        raise TimeoutError('Qt operation timed out: '+window.statusbar.currentMessage())
    def check(name,predicate):
        assert predicate,name
        result['checks'].append({'name':name,'passed':True})
        print('PASS: '+name,flush=True)
    def action(text):
        matches=[a for a in window.findChildren(QAction) if a.text()==text]
        assert len(matches)==1,(text,[a.text() for a in matches]);matches[0].trigger();app.processEvents()
    def button(text):
        b=next(b for b in window.findChildren(QPushButton) if b.text()==text)
        QTest.mouseClick(b,Qt.LeftButton);app.processEvents()
    def export_action(text,path):
        if path.exists():path.unlink()
        controller._choose_save_file=lambda *args,**kwargs:str(path)
        action(text);wait(lambda:path.exists() and not controller._export_in_progress)
    try:
        wav=ROOT/'output_validation/accuracy-audit/signals/vowel.wav'
        duration=sf.info(wav).duration
        times=np.arange(.05,duration-.05,.01);pitch=np.full(len(times),200.);labels=np.full(len(times),2)
        source=out/'vowel_pitch.csv';export_csv(source,times,pitch,controller._praat_default_params(),wav,labels)
        controller._choose_pitch_csv_files=lambda:[str(source)]
        action('Import Pitch CSVs...')
        wait(lambda:controller._loading_entry_index==-1 and len(state.formant_times)>0)
        check('direct CSV import resolves audio without prior audio import',len(controller.batch_entries)==1 and Path(state.audio_path)==wav)
        check('background audio and formant loading completed',state.audio_data is not None and len(state.formant_times)>0)
        np.testing.assert_allclose(state.pitch_values,pitch)
        before=state.snapshot_edit_state()
        window.canvas.region_item.setRegion((.3,.5));window.canvas.region_item.show()
        button('Set Region to Unvoiced')
        mask=(state.timestamps>=.3)&(state.timestamps<=.5)
        check('button->Controller marks unvoiced and clears F0',np.all(state.segment_labels[mask]==1) and np.all(np.isnan(state.pitch_values[mask])))
        window.shortcut_undo.activated.emit();app.processEvents()
        check('undo shortcut signal restores labels and F0',np.array_equal(state.segment_labels,before['segment_labels']) and np.allclose(state.pitch_values,before['pitch_values'],equal_nan=True))
        window.canvas.region_item.setRegion((.7,.9));window.canvas.region_item.show()
        button('Set Region to Silence')
        mask=(state.timestamps>=.7)&(state.timestamps<=.9)
        check('button->Controller marks silence and clears F0',np.all(state.segment_labels[mask]==0) and np.all(np.isnan(state.pitch_values[mask])))
        single=out/'single_acoustic.csv';batch=out/'batch_acoustic.csv';roundtrip=out/'roundtrip_pitch.csv'
        export_action('Export Acoustic Features CSV...',single)
        export_action('Export Batch Acoustic Features CSV...',batch)
        with single.open() as f:a=list(csv.DictReader(f))
        with batch.open() as f:b=list(csv.DictReader(f))
        check('single and batch ExportWorker rows exactly equal',a==b and len(a)==1)
        export_action('Export CSV...',roundtrip)
        expected_t=state.timestamps.copy();expected_f=state.pitch_values.copy();expected_l=state.segment_labels.copy()
        # Change live state after export so cached no-op import cannot falsely pass.
        window.canvas.region_item.setRegion((1.1,1.3));window.canvas.region_item.show()
        button('Set Region to Unvoiced')
        event_count=len(formant_events)
        controller._choose_pitch_csv_files=lambda:[str(roundtrip)]
        action('Import Pitch CSVs...')
        wait(lambda:controller._loading_entry_index==-1 and len(state.formant_times)>0)
        np.testing.assert_allclose(state.timestamps,expected_t,atol=5e-7)
        np.testing.assert_allclose(state.pitch_values,expected_f,equal_nan=True,atol=5e-7)
        np.testing.assert_array_equal(state.segment_labels,expected_l)
        wait(lambda:len(formant_events)>event_count)
        check('export then direct reimport CSV retains track and labels',len(controller.batch_entries)==1)
        window.grab().save(str(out/'annotator.png'))
        result['passed']=True;result['formant_frames']=len(state.formant_times);result['pitch_frames']=len(state.timestamps)
    except Exception:
        result['passed']=False;result['failure']=traceback.format_exc()
        window.grab().save(str(out/'annotator.png'))
    finally:
        controller.cleanup();window.close_handler=lambda:True;window.close();app.processEvents()
        (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2));return result['passed']
if __name__=='__main__':sys.exit(0 if run() else 1)
