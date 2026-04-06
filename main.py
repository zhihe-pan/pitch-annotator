import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from PySide6.QtWidgets import QApplication, QMessageBox, QFileDialog
from PySide6.QtCore import QThread, Signal, QObject, Slot, Qt
from PySide6.QtCore import QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

from ui.main_window import MainWindow
from core.state import PitchState
from core.exporter import export_csv, export_praat_pitch
from backend.audio_core import AudioProcessor
from backend.acoustic_features import export_acoustic_features_csv

class ComputeWorker(QObject):
    # Signals
    finished_loading = Signal(object, object, object, object, int) # S_db, times, freqs, audio_data, sr
    finished_pitch = Signal(object, object, object) # timestamps, pitch_values, segment_labels
    finished_snap = Signal(float, float) # time, snapped_freq
    finished_region_voiced = Signal(float, float, object, object) # region_start, region_end, times, values
    error_occurred = Signal(str)
    
    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        
    @Slot(str)
    def load_audio(self, filepath):
        try:
            self.processor.load_audio(filepath)
            self.finished_loading.emit(
                self.processor.S_db,
                self.processor.spec_times,
                self.processor.spec_freqs,
                self.processor.audio_data,
                int(self.processor.sr),
            )
        except Exception as e:
            self.error_occurred.emit(str(e))
            
    @Slot(float, float, float, float, float, float, float, float)
    def compute_pitch(
        self,
        pitch_floor,
        pitch_ceiling,
        time_step,
        voicing_threshold,
        silence_threshold,
        octave_cost,
        octave_jump_cost,
        voiced_unvoiced_cost,
    ):
        try:
            ts, vals, labels = self.processor.extract_pitch(
                pitch_floor,
                pitch_ceiling,
                time_step,
                voicing_threshold,
                silence_threshold,
                octave_cost,
                octave_jump_cost,
                voiced_unvoiced_cost,
            )
            self.finished_pitch.emit(ts, vals, labels)
        except Exception as e:
            self.error_occurred.emit(str(e))

    @Slot(float, float)
    def snap_point(self, time_val, freq_val):
        try:
            snapped_freq = self.processor.snap_to_peak(time_val, freq_val)
            self.finished_snap.emit(time_val, snapped_freq)
        except Exception as e:
            self.error_occurred.emit(str(e))

    @Slot(float, float, object, float, float, float, float, float, float, float, float)
    def estimate_voiced_region(
        self,
        region_start,
        region_end,
        timestamps,
        pitch_floor,
        pitch_ceiling,
        time_step,
        voicing_threshold,
        silence_threshold,
        octave_cost,
        octave_jump_cost,
        voiced_unvoiced_cost,
    ):
        try:
            times, values = self.processor.estimate_voiced_region(
                region_start,
                region_end,
                timestamps,
                pitch_floor,
                pitch_ceiling,
                time_step,
                voicing_threshold,
                silence_threshold,
                octave_cost,
                octave_jump_cost,
                voiced_unvoiced_cost,
            )
            self.finished_region_voiced.emit(region_start, region_end, times, values)
        except Exception as e:
            self.error_occurred.emit(str(e))

class Controller(QObject):
    request_load = Signal(str)
    request_compute = Signal(float, float, float, float, float, float, float, float)
    request_snap = Signal(float, float)
    request_estimate_voiced_region = Signal(float, float, object, float, float, float, float, float, float, float, float)
    AUTO_PITCH_PRESETS = {
        "NV_female": {"pitch_floor": 100, "pitch_ceiling": 1000},
        "NV_male": {"pitch_floor": 60, "pitch_ceiling": 700},
        "SP_female": {"pitch_floor": 100, "pitch_ceiling": 800},
        "SP_male": {"pitch_floor": 50, "pitch_ceiling": 500},
    }

    def __init__(self, window: MainWindow, state: PitchState, processor: AudioProcessor):
        super().__init__()
        self.window = window
        self.state = state
        self.processor = processor
        self._undo_stack = []
        self._drag_edit_active = False
        
        # Threading setup
        self.thread = QThread()
        self.worker = ComputeWorker(self.processor)
        self.worker.moveToThread(self.thread)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.start()
        self._cleaned_up = False
        
        # Connect thread requests
        self.request_load.connect(self.worker.load_audio, Qt.QueuedConnection)
        self.request_compute.connect(self.worker.compute_pitch, Qt.QueuedConnection)
        self.request_snap.connect(self.worker.snap_point, Qt.QueuedConnection)
        self.request_estimate_voiced_region.connect(self.worker.estimate_voiced_region, Qt.QueuedConnection)
        
        # Connect UI actions
        self.window.open_audio_requested.connect(self._handle_open_audio)
        self.window.export_csv_requested.connect(self._handle_export_csv)
        self.window.export_praat_requested.connect(self._handle_export_praat)
        self.window.export_acoustic_csv_requested.connect(self._handle_export_acoustic_csv)
        self.window.export_all_requested.connect(self._handle_export_all)
        self.window.play_selection_requested.connect(self._handle_play_selection)
        self.window.play_pitch_track_requested.connect(self._handle_play_pitch_track)
        self.window.undo_requested.connect(self._handle_undo)
        
        # Connect Canvas commands
        self.window.canvas.add_point_requested.connect(self._handle_add_point)
        self.window.canvas.remove_point_requested.connect(self._handle_remove_point)
        self.window.canvas.modify_point_requested.connect(self._handle_modify_point)
        self.window.canvas.point_drag_started.connect(self._handle_point_drag_started)
        self.window.canvas.point_drag_finished.connect(self._handle_point_drag_finished)
        self.window.canvas.selection_changed.connect(self._handle_selection_changed)
        
        # Connect Control panel commands
        self.window.control_panel.recompute_requested.connect(self._handle_recompute)
        self.window.control_panel.region_toggled.connect(self.window.canvas.show_region)
        self.window.control_panel.set_region_voiced.connect(self._handle_set_region_voiced)
        self.window.control_panel.set_region_unvoiced.connect(self._handle_set_region_unvoiced)
        self.window.control_panel.set_region_silence.connect(self._handle_set_region_silence)
        self.window.control_panel.volume_changed.connect(self._handle_volume_changed)
        
        # Connect Worker Signals to Controller
        self.worker.finished_loading.connect(self._on_loading_finished, Qt.QueuedConnection)
        self.worker.finished_pitch.connect(self._on_pitch_finished, Qt.QueuedConnection)
        self.worker.finished_snap.connect(self._on_snap_finished, Qt.QueuedConnection)
        self.worker.finished_region_voiced.connect(self._on_region_voiced_estimated, Qt.QueuedConnection)
        self.worker.error_occurred.connect(self._on_error, Qt.QueuedConnection)
        
        # Connect State updates to UI
        self.state.register_callback(self._on_state_changed)
        self.audio_output = QAudioOutput(self.window)
        self.audio_output.setVolume(1.0)
        self.player = QMediaPlayer(self.window)
        self.player.setAudioOutput(self.audio_output)
        self._clip_temp_path = None
        
    def _handle_open_audio(self):
        filepath = self._choose_open_audio_file()
        if filepath:
            self.window.statusbar.showMessage("Loading audio...")
            self.state.audio_path = filepath
            self._apply_pitch_preset_from_filename(filepath)
            self.request_load.emit(filepath)

    def _on_loading_finished(self, S_db, times, freqs, audio_data, sr):
        self.state.set_audio_data(audio_data, sr)
        self._undo_stack.clear()
        self.window.canvas.update_spectrogram(S_db, times, freqs)
        self.window.canvas.fit_to_audio()
        self.window.control_panel.btn_toggle_region.setChecked(True)
        self.window.update_durations(float(times[-1]) if len(times) else 0.0, self.window.canvas.get_selected_region()[1] - self.window.canvas.get_selected_region()[0])
        self.window.statusbar.showMessage("Audio loaded. Computing initial pitch...")
        self._handle_recompute()

    def _handle_recompute(self):
        floor = self.window.control_panel.spin_floor.value()
        ceiling = self.window.control_panel.spin_ceiling.value()
        step = self.window.control_panel.spin_step.value()
        voicing_threshold = self.window.control_panel.spin_voicing_threshold.value()
        silence_threshold = self.window.control_panel.spin_silence_threshold.value()
        octave_cost = self.window.control_panel.spin_octave_cost.value()
        octave_jump_cost = self.window.control_panel.spin_octave_jump_cost.value()
        voiced_unvoiced_cost = self.window.control_panel.spin_voiced_unvoiced_cost.value()

        self.state.pitch_floor = float(floor)
        self.state.pitch_ceiling = float(ceiling)
        self.state.time_step = float(step)
        self.state.voicing_threshold = float(voicing_threshold)
        self.state.silence_threshold = float(silence_threshold)
        self.state.octave_cost = float(octave_cost)
        self.state.octave_jump_cost = float(octave_jump_cost)
        self.state.voiced_unvoiced_cost = float(voiced_unvoiced_cost)
        
        self.window.statusbar.showMessage("Computing pitch...")
        self.request_compute.emit(
            floor,
            ceiling,
            step,
            voicing_threshold,
            silence_threshold,
            octave_cost,
            octave_jump_cost,
            voiced_unvoiced_cost,
        )

    def _on_pitch_finished(self, timestamps, pitch_values, segment_labels):
        self.state.update_pitch_data(timestamps, pitch_values, segment_labels)
        self.window.statusbar.showMessage("Pitch computing finished.", 3000)

    def _on_state_changed(self):
        # Update canvas
        self.window.canvas.update_pitch(self.state.timestamps, self.state.pitch_values)
        self.window.canvas.update_segments(self.state.timestamps, self.state.segment_labels)
        # Update stats
        p20, p50, p80 = self.state.get_quantiles()
        self.window.update_stats(p20, p50, p80, self.state.voice_percent)
        self.window.canvas.update_quantile_lines(p20, p50, p80)

    def _handle_add_point(self, time_val, freq_val):
        if self.state.audio_path is None:
            return
        self._push_undo_state()
        self.request_snap.emit(time_val, freq_val)

    def _on_snap_finished(self, time_val, snapped_freq):
        self.state.add_or_update_point(time_val, snapped_freq)

    def _handle_remove_point(self, time_val):
        self._push_undo_state()
        self.state.remove_point(time_val)

    def _handle_modify_point(self, time_val, freq_val):
        if self.state.audio_path is None:
            return
        self.state.add_or_update_point(time_val, freq_val)

    def _handle_point_drag_started(self):
        if not self._drag_edit_active:
            self._push_undo_state()
            self._drag_edit_active = True

    def _handle_point_drag_finished(self):
        self._drag_edit_active = False

    def _handle_set_region_voiced(self):
        r_min, r_max = self.window.canvas.get_selected_region()
        if len(self.state.timestamps) == 0:
            return
        self._push_undo_state()
        self.window.statusbar.showMessage("Estimating voiced region pitch...")
        self.request_estimate_voiced_region.emit(
            r_min,
            r_max,
            self.state.timestamps,
            self.state.pitch_floor,
            self.state.pitch_ceiling,
            self.state.time_step,
            self.state.voicing_threshold,
            self.state.silence_threshold,
            self.state.octave_cost,
            self.state.octave_jump_cost,
            self.state.voiced_unvoiced_cost,
        )

    def _on_region_voiced_estimated(self, region_start, region_end, region_times, region_values):
        if len(region_times) == 0:
            self.window.statusbar.showMessage("No frames found in selected region.", 3000)
            return
        self.state.set_voiced(region_start, region_end, region_values)
        self.window.statusbar.showMessage(
            f"Selected region set to voiced. Voice%: {self.state.voice_percent:.1f}",
            3000,
        )

    def _handle_set_region_unvoiced(self):
        r_min, r_max = self.window.canvas.get_selected_region()
        self._push_undo_state()
        self.state.set_unvoiced(r_min, r_max)
        self.window.statusbar.showMessage(
            f"Selected region set to unvoiced. Voice%: {self.state.voice_percent:.1f}",
            3000,
        )

    def _handle_set_region_silence(self):
        r_min, r_max = self.window.canvas.get_selected_region()
        self._push_undo_state()
        self.state.set_silence(r_min, r_max)
        self.window.statusbar.showMessage(
            f"Selected region set to silence. Voice%: {self.state.voice_percent:.1f}",
            3000,
        )

    def _handle_volume_changed(self, value):
        self.audio_output.setVolume(max(0.0, min(1.0, value / 100.0)))

    def _handle_selection_changed(self, start_time, end_time):
        total_duration = float(self.state.audio_data.shape[0] / self.state.sample_rate) if self.state.audio_data is not None and self.state.sample_rate else 0.0
        self.window.update_durations(total_duration, max(0.0, end_time - start_time))

    def _push_undo_state(self):
        if len(self.state.pitch_values) == 0:
            return
        snapshot = self.state.snapshot_edit_state()
        self._undo_stack.append(snapshot)
        self._last_snapshot = snapshot
        if len(self._undo_stack) > 100:
            self._undo_stack.pop(0)

    def _handle_undo(self):
        if not self._undo_stack:
            self.window.statusbar.showMessage("Nothing to undo.", 2000)
            return
        snapshot = self._undo_stack.pop()
        self.state.restore_edit_state(snapshot)
        self._drag_edit_active = False
        self.window.statusbar.showMessage("Undid last edit.", 2000)

    def _apply_pitch_preset_from_filename(self, filepath):
        name = Path(filepath).stem.lower()
        is_female = "gender2" in name
        is_nv = "_nv_" in f"_{name}_" or name.endswith("_nv")
        preset_key = f"{'NV' if is_nv else 'SP'}_{'female' if is_female else 'male'}"
        preset = self.AUTO_PITCH_PRESETS[preset_key]

        self.window.control_panel.spin_floor.setValue(int(preset["pitch_floor"]))
        self.window.control_panel.spin_ceiling.setValue(int(preset["pitch_ceiling"]))

    def _default_export_path(self, suffix):
        if not self.state.audio_path:
            return ""
        audio_path = Path(self.state.audio_path).resolve()
        return str(audio_path.with_name(f"{audio_path.stem}{suffix}"))

    def _choose_open_audio_file(self):
        start_dir = ""
        if self.state.audio_path:
            start_dir = str(Path(self.state.audio_path).resolve().parent)

        dialog = QFileDialog(self.window, "Open Audio File", start_dir)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setNameFilters(
            [
                "Audio Files (*.wav *.WAV *.mp3 *.MP3 *.m4a *.M4A *.flac *.FLAC *.aiff *.AIFF *.aif *.AIF *.ogg *.OGG)",
                "All Files (*)",
            ]
        )
        if dialog.exec() != QFileDialog.Accepted:
            return ""
        selected = dialog.selectedFiles()
        return selected[0] if selected else ""

    def _choose_save_file(self, title, default_path, file_filter):
        dialog = QFileDialog(self.window, title, default_path)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setNameFilter(file_filter)
        if default_path:
            dialog.selectFile(default_path)
        if dialog.exec() != QFileDialog.Accepted:
            return ""
        selected = dialog.selectedFiles()
        return selected[0] if selected else ""

    def _choose_directory(self, title, start_dir):
        dialog = QFileDialog(self.window, title, start_dir)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        if dialog.exec() != QFileDialog.Accepted:
            return ""
        selected = dialog.selectedFiles()
        return selected[0] if selected else ""

    def _current_pitch_params(self):
        return {
            "pitch_floor": self.state.pitch_floor,
            "pitch_ceiling": self.state.pitch_ceiling,
            "voicing_threshold": self.state.voicing_threshold,
            "silence_threshold": self.state.silence_threshold,
            "octave_cost": self.state.octave_cost,
            "octave_jump_cost": self.state.octave_jump_cost,
            "voiced_unvoiced_cost": self.state.voiced_unvoiced_cost,
        }

    def _handle_export_csv(self):
        filepath = self._choose_save_file(
            "Export CSV",
            self._default_export_path("_pitch.csv"),
            "CSV Files (*.csv)",
        )
        if filepath:
            export_csv(filepath, self.state.timestamps, self.state.pitch_values)
            self.window.statusbar.showMessage(f"Exported to {filepath}", 3000)

    def _handle_export_praat(self):
        filepath = self._choose_save_file(
            "Export Praat Pitch",
            self._default_export_path(".Pitch"),
            "Pitch Files (*.Pitch)",
        )
        if filepath:
            export_praat_pitch(filepath, self.state.timestamps, self.state.pitch_values)
            self.window.statusbar.showMessage(f"Exported to {filepath}", 3000)

    def _handle_export_acoustic_csv(self):
        if not self.state.audio_path:
            self.window.statusbar.showMessage("No audio loaded.", 3000)
            return
        filepath = self._choose_save_file(
            "Export Acoustic Features CSV",
            self._default_export_path("_acoustic_features.csv"),
            "CSV Files (*.csv)",
        )
        if filepath:
            export_acoustic_features_csv(self.state.audio_path, filepath, self._current_pitch_params())
            self.window.statusbar.showMessage(f"Exported acoustic features to {filepath}", 3000)

    def _handle_export_all(self):
        if not self.state.audio_path:
            self.window.statusbar.showMessage("No audio loaded.", 3000)
            return
        output_dir = self._choose_directory(
            "Export All",
            str(Path(self.state.audio_path).resolve().parent),
        )
        if not output_dir:
            return

        stem = Path(self.state.audio_path).stem
        base_dir = Path(output_dir)
        pitch_csv_path = base_dir / f"{stem}_pitch.csv"
        praat_path = base_dir / f"{stem}.Pitch"
        acoustic_path = base_dir / f"{stem}_acoustic_features.csv"

        export_csv(str(pitch_csv_path), self.state.timestamps, self.state.pitch_values)
        export_praat_pitch(str(praat_path), self.state.timestamps, self.state.pitch_values)
        export_acoustic_features_csv(self.state.audio_path, str(acoustic_path), self._current_pitch_params())
        self.window.statusbar.showMessage(f"Exported all files to {base_dir}", 4000)

    def _handle_play_selection(self):
        if self.state.audio_data is None or self.state.sample_rate is None:
            self.window.statusbar.showMessage("No audio loaded.", 3000)
            return

        start_time, end_time = self.window.canvas.get_selected_region()
        if end_time <= start_time:
            self.window.statusbar.showMessage("Selected region is empty.", 3000)
            return

        sr = int(self.state.sample_rate)
        start_idx = max(0, int(np.floor(start_time * sr)))
        end_idx = min(len(self.state.audio_data), int(np.ceil(end_time * sr)))
        if end_idx <= start_idx:
            self.window.statusbar.showMessage("Selected region is empty.", 3000)
            return

        clip = np.asarray(self.state.audio_data[start_idx:end_idx], dtype=np.float32)
        self._write_temp_clip(clip, sr)
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(self._clip_temp_path))
        self.player.play()
        self.window.statusbar.showMessage(
            f"Playing selection {start_time:.3f}s - {end_time:.3f}s",
            3000,
        )

    def _handle_play_pitch_track(self):
        if self.state.audio_data is None or self.state.sample_rate is None:
            self.window.statusbar.showMessage("No audio loaded.", 3000)
            return
        if len(self.state.timestamps) == 0 or len(self.state.pitch_values) == 0:
            self.window.statusbar.showMessage("No pitch track available.", 3000)
            return

        start_time, end_time = self.window.canvas.get_selected_region()
        if end_time <= start_time:
            self.window.statusbar.showMessage("Selected region is empty.", 3000)
            return

        sr = int(self.state.sample_rate)
        clip = self._synthesize_pitch_clip(start_time, end_time, sr)
        if clip is None or len(clip) == 0:
            self.window.statusbar.showMessage("No voiced pitch data in selected region.", 3000)
            return

        self._write_temp_clip(clip, sr)
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(self._clip_temp_path))
        self.player.play()
        self.window.statusbar.showMessage(
            f"Playing pitch track {start_time:.3f}s - {end_time:.3f}s",
            3000,
        )

    def _synthesize_pitch_clip(self, start_time, end_time, sample_rate):
        duration = max(0.0, float(end_time - start_time))
        if duration <= 0.0:
            return None

        frame_times = np.asarray(self.state.timestamps, dtype=float)
        frame_freqs = np.asarray(self.state.pitch_values, dtype=float)
        if len(frame_times) == 0 or len(frame_freqs) == 0:
            return None

        sample_count = max(1, int(np.ceil(duration * sample_rate)))
        sample_times = start_time + np.arange(sample_count, dtype=float) / float(sample_rate)

        voiced_frame_mask = (~np.isnan(frame_freqs)) & (frame_freqs > 0)
        if len(self.state.segment_labels) == len(frame_freqs):
            voiced_frame_mask &= (np.asarray(self.state.segment_labels, dtype=int) == 2)

        if not np.any(voiced_frame_mask):
            return None

        freq_samples = np.interp(
            sample_times,
            frame_times,
            np.where(voiced_frame_mask, frame_freqs, 0.0),
            left=0.0,
            right=0.0,
        )
        amp_samples = np.interp(
            sample_times,
            frame_times,
            np.where(voiced_frame_mask, 1.0, 0.0),
            left=0.0,
            right=0.0,
        )
        amp_samples = (amp_samples > 0.5).astype(np.float32)
        if not np.any(amp_samples):
            return None

        phase = 2.0 * np.pi * np.cumsum(freq_samples) / float(sample_rate)
        waveform = 0.18 * np.sin(phase)

        fade_samples = max(1, min(sample_rate // 200, sample_count // 20))
        if fade_samples > 1:
            fade = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
            amp_samples[:fade_samples] *= fade
            amp_samples[-fade_samples:] *= fade[::-1]

        clip = waveform.astype(np.float32) * amp_samples
        return clip

    def _write_temp_clip(self, clip, sr):
        if self._clip_temp_path:
            try:
                Path(self._clip_temp_path).unlink(missing_ok=True)
            except Exception:
                pass

        handle = tempfile.NamedTemporaryFile(prefix="pitch_annotator_clip_", suffix=".wav", delete=False)
        handle.close()
        sf.write(handle.name, clip, sr)
        self._clip_temp_path = handle.name

    def _on_error(self, err_msg):
        QMessageBox.critical(self.window, "Error", err_msg)
        self.window.statusbar.showMessage("Error occurred.")

    def cleanup(self):
        if self._cleaned_up:
            return
        self._cleaned_up = True
        self.player.stop()
        self.thread.quit()
        self.thread.wait()
        self.thread.deleteLater()
        if self._clip_temp_path:
            try:
                Path(self._clip_temp_path).unlink(missing_ok=True)
            except Exception:
                pass

def main():
    app = QApplication(sys.argv)
    
    window = MainWindow()
    state = PitchState()
    processor = AudioProcessor()
    
    controller = Controller(window, state, processor)
    app.aboutToQuit.connect(controller.cleanup)
    
    window.show()
    window.raise_()
    window.activateWindow()
    
    out = app.exec()
    controller.cleanup()
    sys.exit(out)

if __name__ == "__main__":
    main()
