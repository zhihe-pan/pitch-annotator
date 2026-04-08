import csv
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot, QStandardPaths, QByteArray, QBuffer, QIODevice, QUrl
from PySide6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QDialog

from backend.acoustic_features import (
    compute_feature_row_with_pitch_overrides,
    export_acoustic_features_csv,
)
from backend.audio_core import AudioProcessor
from core.exporter import PARAMETER_COLUMNS, export_csv, export_praat_pitch
from core.state import PitchState
from ui.batch_import_dialog import BatchImportDialog
from ui.main_window import MainWindow


@dataclass
class BatchAudioEntry:
    filepath: str
    params: dict
    state_snapshot: dict | None = None
    spectrogram_cache: dict | None = None
    selection_region: tuple[float, float] = (0.0, 0.25)
    region_visible: bool = True
    acoustic_row: dict | None = None
    dirty: bool = False


class ComputeWorker(QObject):
    finished_loading = Signal(int, str, object, object, object, object, int)
    finished_pitch = Signal(int, str, object, object, object, object, object, object, object)
    finished_snap = Signal(int, str, float, float)
    finished_region_voiced = Signal(int, str, float, float, object, object)
    error_occurred = Signal(str)

    def __init__(self, processor):
        super().__init__()
        self.processor = processor

    @Slot(int, str)
    def load_audio(self, entry_index, filepath):
        try:
            self.processor.load_audio(filepath)
            self.finished_loading.emit(
                int(entry_index),
                str(filepath),
                self.processor.S_db,
                self.processor.spec_times,
                self.processor.spec_freqs,
                self.processor.audio_data,
                int(self.processor.sr),
            )
        except Exception as e:
            self.error_occurred.emit(str(e))

    @Slot(int, str, float, float, float, float, float, float, float, float)
    def compute_pitch(
        self,
        entry_index,
        filepath,
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
            if self.processor.loaded_filepath != str(filepath):
                self.processor.load_audio(filepath)
            ts, vals, labels, formant_times, f1_values, f2_values, f3_values = self.processor.extract_pitch(
                pitch_floor,
                pitch_ceiling,
                time_step,
                voicing_threshold,
                silence_threshold,
                octave_cost,
                octave_jump_cost,
                voiced_unvoiced_cost,
            )
            self.finished_pitch.emit(
                int(entry_index),
                str(filepath),
                ts,
                vals,
                labels,
                formant_times,
                f1_values,
                f2_values,
                f3_values,
            )
        except Exception as e:
            self.error_occurred.emit(str(e))

    @Slot(int, str, float, float)
    def snap_point(self, entry_index, filepath, time_val, freq_val):
        try:
            if self.processor.loaded_filepath != str(filepath):
                self.processor.load_audio(filepath)
            snapped_freq = self.processor.snap_to_peak(time_val, freq_val)
            self.finished_snap.emit(int(entry_index), str(filepath), time_val, snapped_freq)
        except Exception as e:
            self.error_occurred.emit(str(e))

    @Slot(int, str, float, float, object, float, float, float, float, float, float, float, float)
    def estimate_voiced_region(
        self,
        entry_index,
        filepath,
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
            if self.processor.loaded_filepath != str(filepath):
                self.processor.load_audio(filepath)
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
            self.finished_region_voiced.emit(int(entry_index), str(filepath), region_start, region_end, times, values)
        except Exception as e:
            self.error_occurred.emit(str(e))


class ExportWorker(QObject):
    finished = Signal(str)
    error_occurred = Signal(str)

    @Slot(object)
    def run_task(self, task):
        try:
            task_type = task["type"]
            if task_type == "pitch_csv":
                export_csv(
                    task["filepath"],
                    task["timestamps"],
                    task["pitch_values"],
                    pitch_params=task["pitch_params"],
                    audio_path=task["audio_path"],
                    segment_labels=task["segment_labels"],
                )
                self.finished.emit(f"Exported to {task['filepath']}")
                return

            if task_type == "praat_pitch":
                export_praat_pitch(task["filepath"], task["timestamps"], task["pitch_values"])
                self.finished.emit(f"Exported to {task['filepath']}")
                return

            if task_type == "acoustic_csv":
                row = compute_feature_row_with_pitch_overrides(
                    task["audio_path"],
                    task["pitch_params"],
                    task["timestamps"],
                    task["pitch_values"],
                    task["segment_labels"],
                )
                with open(task["filepath"], "w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
                    writer.writeheader()
                    writer.writerow(row)
                self.finished.emit(f"Exported acoustic features to {task['filepath']}")
                return

            if task_type == "batch_acoustic_csv":
                rows = []
                for entry in task["entries"]:
                    row = compute_feature_row_with_pitch_overrides(
                        entry["audio_path"],
                        entry["pitch_params"],
                        entry["timestamps"],
                        entry["pitch_values"],
                        entry["segment_labels"],
                    )
                    rows.append(row)

                preferred = ["audio_file"] + PARAMETER_COLUMNS
                fieldnames = list(preferred)
                for row in rows:
                    for key in row.keys():
                        if key not in fieldnames:
                            fieldnames.append(key)
                with open(task["filepath"], "w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                self.finished.emit(f"Exported batch acoustic features to {task['filepath']}")
                return

            if task_type == "export_all":
                export_csv(
                    task["pitch_csv_path"],
                    task["timestamps"],
                    task["pitch_values"],
                    pitch_params=task["pitch_params"],
                    audio_path=task["audio_path"],
                    segment_labels=task["segment_labels"],
                )
                export_praat_pitch(task["praat_path"], task["timestamps"], task["pitch_values"])
                row = compute_feature_row_with_pitch_overrides(
                    task["audio_path"],
                    task["pitch_params"],
                    task["timestamps"],
                    task["pitch_values"],
                    task["segment_labels"],
                )
                with open(task["acoustic_csv_path"], "w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
                    writer.writeheader()
                    writer.writerow(row)
                self.finished.emit(f"Exported all files to {task['output_dir']}")
                return

            raise ValueError(f"Unknown export task type: {task_type}")
        except Exception as e:
            self.error_occurred.emit(str(e))


class Controller(QObject):
    request_load = Signal(int, str)
    request_compute = Signal(int, str, float, float, float, float, float, float, float, float)
    request_snap = Signal(int, str, float, float)
    request_estimate_voiced_region = Signal(int, str, float, float, object, float, float, float, float, float, float, float, float)
    request_export = Signal(object)

    def __init__(self, window: MainWindow, state: PitchState, processor: AudioProcessor):
        super().__init__()
        self.window = window
        self.state = state
        self.processor = processor
        self.batch_entries: list[BatchAudioEntry] = []
        self.current_entry_index = -1
        self._loading_entry_index = -1
        self._undo_stack = []
        self._drag_edit_active = False
        self._cleaned_up = False
        self._suppress_dirty_tracking = False
        self._audio_sink = None
        self._audio_buffer = None
        self._playback_volume = 1.0
        self._audio_output_devices = []
        self._selected_audio_output_index = 0

        self.thread = QThread()
        self.worker = ComputeWorker(self.processor)
        self.worker.moveToThread(self.thread)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.start()

        self.request_load.connect(self.worker.load_audio, Qt.QueuedConnection)
        self.request_compute.connect(self.worker.compute_pitch, Qt.QueuedConnection)
        self.request_snap.connect(self.worker.snap_point, Qt.QueuedConnection)
        self.request_estimate_voiced_region.connect(self.worker.estimate_voiced_region, Qt.QueuedConnection)

        self.export_thread = QThread()
        self.export_worker = ExportWorker()
        self.export_worker.moveToThread(self.export_thread)
        self.export_thread.finished.connect(self.export_worker.deleteLater)
        self.export_thread.start()
        self.request_export.connect(self.export_worker.run_task, Qt.QueuedConnection)

        self.window.open_audio_requested.connect(self._handle_open_audio)
        self.window.current_audio_index_changed.connect(self._handle_audio_index_changed)
        self.window.next_audio_requested.connect(self._handle_next_audio)
        self.window.previous_audio_requested.connect(self._handle_previous_audio)
        self.window.export_csv_requested.connect(self._handle_export_csv)
        self.window.export_praat_requested.connect(self._handle_export_praat)
        self.window.export_acoustic_csv_requested.connect(self._handle_export_acoustic_csv)
        self.window.batch_export_acoustic_csv_requested.connect(self._handle_batch_export_acoustic_csv)
        self.window.export_all_requested.connect(self._handle_export_all)
        self.window.play_selection_requested.connect(self._handle_play_selection)
        self.window.play_pitch_track_requested.connect(self._handle_play_pitch_track)
        self.window.undo_requested.connect(self._handle_undo)

        self.window.canvas.add_point_requested.connect(self._handle_add_point)
        self.window.canvas.remove_point_requested.connect(self._handle_remove_point)
        self.window.canvas.modify_point_requested.connect(self._handle_modify_point)
        self.window.canvas.point_drag_started.connect(self._handle_point_drag_started)
        self.window.canvas.point_drag_finished.connect(self._handle_point_drag_finished)
        self.window.canvas.selection_changed.connect(self._handle_selection_changed)

        self.window.control_panel.recompute_requested.connect(self._handle_recompute)
        self.window.control_panel.region_toggled.connect(self._handle_region_toggled)
        self.window.control_panel.set_region_voiced.connect(self._handle_set_region_voiced)
        self.window.control_panel.set_region_unvoiced.connect(self._handle_set_region_unvoiced)
        self.window.control_panel.set_region_silence.connect(self._handle_set_region_silence)
        self.window.control_panel.volume_changed.connect(self._handle_volume_changed)
        self.window.control_panel.audio_output_device_changed.connect(self._handle_audio_output_device_changed)

        self.worker.finished_loading.connect(self._on_loading_finished, Qt.QueuedConnection)
        self.worker.finished_pitch.connect(self._on_pitch_finished, Qt.QueuedConnection)
        self.worker.finished_snap.connect(self._on_snap_finished, Qt.QueuedConnection)
        self.worker.finished_region_voiced.connect(self._on_region_voiced_estimated, Qt.QueuedConnection)
        self.worker.error_occurred.connect(self._on_error, Qt.QueuedConnection)
        self.export_worker.finished.connect(self._on_export_finished, Qt.QueuedConnection)
        self.export_worker.error_occurred.connect(self._on_export_error, Qt.QueuedConnection)

        self.state.register_callback(self._on_state_changed)
        self._refresh_audio_output_devices()
        self._export_in_progress = False

    def _praat_default_params(self):
        return {
            "pitch_floor": 50.0,
            "pitch_ceiling": 800.0,
            "time_step": 0.0,
            "voicing_threshold": 0.50,
            "silence_threshold": 0.09,
            "octave_cost": 0.055,
            "octave_jump_cost": 0.35,
            "voiced_unvoiced_cost": 0.14,
        }

    def _handle_open_audio(self):
        filepaths = self._choose_open_audio_files()
        if not filepaths:
            return
        params = self._ask_batch_import_params()
        if params is None:
            return

        self._save_current_entry_state()
        existing_paths = {entry.filepath for entry in self.batch_entries}
        new_paths = [path for path in filepaths if path not in existing_paths]
        if not new_paths:
            self.window.statusbar.showMessage("All selected audio files are already in the list.", 3000)
            return

        first_new_index = len(self.batch_entries)
        self.batch_entries.extend(BatchAudioEntry(filepath=path, params=dict(params)) for path in new_paths)
        self.window.set_audio_files([entry.filepath for entry in self.batch_entries])
        self.window.set_current_audio_index(first_new_index)
        self._switch_to_entry(first_new_index)

    def _choose_open_audio_files(self):
        start_dir = ""
        if self.current_entry_index >= 0 and self.batch_entries:
            start_dir = str(Path(self.batch_entries[self.current_entry_index].filepath).resolve().parent)

        dialog = QFileDialog(self.window, "Import Audio Files", start_dir)
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        self._configure_file_dialog(dialog, start_dir)
        dialog.setNameFilters(
            [
                "Audio Files (*.wav *.WAV *.mp3 *.MP3 *.m4a *.M4A *.flac *.FLAC *.aiff *.AIFF *.aif *.AIF *.ogg *.OGG)",
                "All Files (*)",
            ]
        )
        if dialog.exec() != QFileDialog.Accepted:
            return []
        return dialog.selectedFiles()

    def _ask_batch_import_params(self):
        dialog = BatchImportDialog(self.window)
        defaults = self._praat_default_params()
        dialog.spin_floor.setValue(int(defaults["pitch_floor"]))
        dialog.spin_ceiling.setValue(int(defaults["pitch_ceiling"]))
        dialog.spin_step.setValue(float(defaults["time_step"]))
        dialog.spin_voicing_threshold.setValue(float(defaults["voicing_threshold"]))
        dialog.spin_silence_threshold.setValue(float(defaults["silence_threshold"]))
        dialog.spin_octave_cost.setValue(float(defaults["octave_cost"]))
        dialog.spin_octave_jump_cost.setValue(float(defaults["octave_jump_cost"]))
        dialog.spin_voiced_unvoiced_cost.setValue(float(defaults["voiced_unvoiced_cost"]))
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.get_params()

    def _handle_audio_index_changed(self, index):
        if 0 <= index < len(self.batch_entries) and index != self.current_entry_index:
            self._switch_to_entry(index)

    def _handle_next_audio(self):
        if not self.batch_entries:
            return
        next_index = min(self.current_entry_index + 1, len(self.batch_entries) - 1)
        self.window.set_current_audio_index(next_index)
        self._switch_to_entry(next_index)

    def _handle_previous_audio(self):
        if not self.batch_entries:
            return
        prev_index = max(self.current_entry_index - 1, 0)
        self.window.set_current_audio_index(prev_index)
        self._switch_to_entry(prev_index)

    def _switch_to_entry(self, index):
        if index < 0 or index >= len(self.batch_entries):
            return
        self._save_current_entry_state()
        self.current_entry_index = index
        self.window.set_current_audio_index(index)
        entry = self.batch_entries[index]
        self._undo_stack.clear()
        self._apply_params_to_controls(entry.params)

        if entry.state_snapshot is not None and entry.spectrogram_cache is not None:
            self._suppress_dirty_tracking = True
            self.state.restore_full_state(entry.state_snapshot)
            self.window.canvas.update_spectrogram(
                entry.spectrogram_cache["S_db"],
                entry.spectrogram_cache["times"],
                entry.spectrogram_cache["freqs"],
            )
            self.window.canvas.fit_to_audio()
            self.window.canvas.region_item.setRegion(list(entry.selection_region))
            self.window.canvas.show_region(entry.region_visible)
            self.window.update_durations(
                float(self.state.audio_data.shape[0] / self.state.sample_rate) if self.state.audio_data is not None and self.state.sample_rate else 0.0,
                max(0.0, entry.selection_region[1] - entry.selection_region[0]) if entry.region_visible else 0.0,
            )
            self._suppress_dirty_tracking = False
            self.window.statusbar.showMessage(f"Loaded cached audio: {Path(entry.filepath).name}", 2000)
            return

        self._loading_entry_index = index
        self.state.audio_path = entry.filepath
        self.window.statusbar.showMessage(f"Loading {Path(entry.filepath).name}...")
        self.request_load.emit(index, entry.filepath)

    def _on_loading_finished(self, entry_index, filepath, S_db, times, freqs, audio_data, sr):
        if entry_index < 0 or entry_index >= len(self.batch_entries):
            return
        entry = self.batch_entries[entry_index]
        if entry.filepath != filepath:
            return
        entry.spectrogram_cache = {"S_db": S_db, "times": times, "freqs": freqs}
        if entry_index != self.current_entry_index:
            return
        self.state.reset()
        self.state.audio_path = entry.filepath
        self.state.set_audio_data(audio_data, sr)
        self.window.canvas.update_spectrogram(S_db, times, freqs)
        self.window.canvas.fit_to_audio()
        self.window.canvas.region_item.setRegion(list(entry.selection_region))
        self.window.canvas.show_region(entry.region_visible)
        self.window.update_durations(
            float(times[-1]) if len(times) else 0.0,
            max(0.0, entry.selection_region[1] - entry.selection_region[0]) if entry.region_visible else 0.0,
        )
        self.window.statusbar.showMessage("Audio loaded. Computing initial pitch...")
        self._handle_recompute()

    def _handle_recompute(self):
        if self.current_entry_index < 0 or self.current_entry_index >= len(self.batch_entries):
            return
        floor = float(self.window.control_panel.spin_floor.value())
        ceiling = float(self.window.control_panel.spin_ceiling.value())
        step = float(self.window.control_panel.spin_step.value())
        voicing_threshold = float(self.window.control_panel.spin_voicing_threshold.value())
        silence_threshold = float(self.window.control_panel.spin_silence_threshold.value())
        octave_cost = float(self.window.control_panel.spin_octave_cost.value())
        octave_jump_cost = float(self.window.control_panel.spin_octave_jump_cost.value())
        voiced_unvoiced_cost = float(self.window.control_panel.spin_voiced_unvoiced_cost.value())

        self.state.pitch_floor = floor
        self.state.pitch_ceiling = ceiling
        self.state.time_step = step
        self.state.voicing_threshold = voicing_threshold
        self.state.silence_threshold = silence_threshold
        self.state.octave_cost = octave_cost
        self.state.octave_jump_cost = octave_jump_cost
        self.state.voiced_unvoiced_cost = voiced_unvoiced_cost
        self.batch_entries[self.current_entry_index].params = self._current_pitch_params()
        self.batch_entries[self.current_entry_index].acoustic_row = None

        self.window.statusbar.showMessage("Computing pitch...")
        self.request_compute.emit(
            self.current_entry_index,
            self.batch_entries[self.current_entry_index].filepath,
            floor,
            ceiling,
            step,
            voicing_threshold,
            silence_threshold,
            octave_cost,
            octave_jump_cost,
            voiced_unvoiced_cost,
        )

    def _on_pitch_finished(self, entry_index, filepath, timestamps, pitch_values, segment_labels, formant_times, f1_values, f2_values, f3_values):
        if entry_index < 0 or entry_index >= len(self.batch_entries):
            return
        entry = self.batch_entries[entry_index]
        if entry.filepath != filepath or entry_index != self.current_entry_index:
            return
        self.state.update_pitch_data(
            timestamps,
            pitch_values,
            segment_labels,
            formant_times,
            f1_values,
            f2_values,
            f3_values,
        )
        self._save_current_entry_state()
        self.window.statusbar.showMessage("Pitch computing finished.", 3000)
        self._loading_entry_index = -1
        self._suppress_dirty_tracking = False

    def _save_current_entry_state(self):
        if self.current_entry_index < 0 or self.current_entry_index >= len(self.batch_entries):
            return
        entry = self.batch_entries[self.current_entry_index]
        if self.state.audio_data is None or self.state.sample_rate is None:
            return
        entry.state_snapshot = self.state.snapshot_full_state()
        start, end = self.window.canvas.get_selected_region()
        entry.selection_region = (float(start), float(end))
        entry.region_visible = self.window.canvas.region_item.isVisible()
        entry.params = self._current_pitch_params()
        entry.acoustic_row = None

    def _apply_params_to_controls(self, params):
        self.window.control_panel.spin_floor.setValue(int(params["pitch_floor"]))
        self.window.control_panel.spin_ceiling.setValue(int(params["pitch_ceiling"]))
        self.window.control_panel.spin_step.setValue(float(params["time_step"]))
        self.window.control_panel.spin_voicing_threshold.setValue(float(params["voicing_threshold"]))
        self.window.control_panel.spin_silence_threshold.setValue(float(params["silence_threshold"]))
        self.window.control_panel.spin_octave_cost.setValue(float(params["octave_cost"]))
        self.window.control_panel.spin_octave_jump_cost.setValue(float(params["octave_jump_cost"]))
        self.window.control_panel.spin_voiced_unvoiced_cost.setValue(float(params["voiced_unvoiced_cost"]))

    def _on_state_changed(self):
        self.window.canvas.update_pitch(self.state.timestamps, self.state.pitch_values)
        self.window.canvas.update_segments(self.state.timestamps, self.state.segment_labels)
        self.window.canvas.update_formants(
            self.state.formant_times,
            self.state.f1_values,
            self.state.f2_values,
            self.state.f3_values,
        )
        p20, p50, p80 = self.state.get_quantiles()
        self.window.update_stats(p20, p50, p80, self.state.voice_percent)
        self.window.canvas.update_quantile_lines(p20, p50, p80)
        if self.current_entry_index >= 0 and self.current_entry_index < len(self.batch_entries):
            self._save_current_entry_state()

    def _refresh_formants_from_state(self):
        if self.state.audio_data is None or self.state.sample_rate is None:
            return
        formant_times, f1_values, f2_values, f3_values = self.processor.extract_formants_for_track(
            self.state.timestamps,
            self.state.pitch_values,
            self.state.segment_labels,
        )
        self.state.update_formant_data(formant_times, f1_values, f2_values, f3_values)

    def _handle_add_point(self, time_val, freq_val):
        if self.state.audio_path is None:
            return
        self._push_undo_state()
        self.request_snap.emit(self.current_entry_index, self.batch_entries[self.current_entry_index].filepath, time_val, freq_val)

    def _on_snap_finished(self, entry_index, filepath, time_val, snapped_freq):
        if entry_index != self.current_entry_index or self.state.audio_path != filepath:
            return
        self.state.add_or_update_point(time_val, snapped_freq)
        self._refresh_formants_from_state()
        self._mark_current_entry_dirty()

    def _handle_remove_point(self, time_val):
        self._push_undo_state()
        self.state.remove_point(time_val)
        self._refresh_formants_from_state()
        self._mark_current_entry_dirty()

    def _handle_modify_point(self, time_val, freq_val):
        if self.state.audio_path is None:
            return
        self.state.add_or_update_point(time_val, freq_val)
        self._refresh_formants_from_state()
        self._mark_current_entry_dirty()

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
        self._mark_current_entry_dirty()
        self.window.statusbar.showMessage("Estimating voiced region pitch...")
        self.request_estimate_voiced_region.emit(
            self.current_entry_index,
            self.batch_entries[self.current_entry_index].filepath,
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

    def _on_region_voiced_estimated(self, entry_index, filepath, region_start, region_end, region_times, region_values):
        if entry_index != self.current_entry_index or self.state.audio_path != filepath:
            return
        if len(region_times) == 0:
            self.window.statusbar.showMessage("No frames found in selected region.", 3000)
            return
        self.state.set_voiced(region_start, region_end, region_values)
        self._refresh_formants_from_state()
        self.window.statusbar.showMessage(f"Selected region set to voiced. Voice%: {self.state.voice_percent:.1f}", 3000)

    def _handle_set_region_unvoiced(self):
        r_min, r_max = self.window.canvas.get_selected_region()
        self._push_undo_state()
        self.state.set_unvoiced(r_min, r_max)
        self._refresh_formants_from_state()
        self._mark_current_entry_dirty()
        self.window.statusbar.showMessage(f"Selected region set to unvoiced. Voice%: {self.state.voice_percent:.1f}", 3000)

    def _handle_set_region_silence(self):
        r_min, r_max = self.window.canvas.get_selected_region()
        self._push_undo_state()
        self.state.set_silence(r_min, r_max)
        self._refresh_formants_from_state()
        self._mark_current_entry_dirty()
        self.window.statusbar.showMessage(f"Selected region set to silence. Voice%: {self.state.voice_percent:.1f}", 3000)

    def _handle_selection_changed(self, start_time, end_time):
        total_duration = float(self.state.audio_data.shape[0] / self.state.sample_rate) if self.state.audio_data is not None and self.state.sample_rate else 0.0
        selection_duration = max(0.0, end_time - start_time) if self.window.canvas.region_item.isVisible() else 0.0
        self.window.update_durations(total_duration, selection_duration)
        if 0 <= self.current_entry_index < len(self.batch_entries):
            self.batch_entries[self.current_entry_index].selection_region = (float(start_time), float(end_time))
            self.batch_entries[self.current_entry_index].region_visible = self.window.canvas.region_item.isVisible()

    def _handle_region_toggled(self, show):
        self.window.canvas.show_region(show)
        if 0 <= self.current_entry_index < len(self.batch_entries):
            self.batch_entries[self.current_entry_index].region_visible = bool(show)
        start, end = self.window.canvas.get_selected_region()
        self._handle_selection_changed(start, end)

    def _push_undo_state(self):
        if len(self.state.pitch_values) == 0:
            return
        self._undo_stack.append(self.state.snapshot_edit_state())
        if len(self._undo_stack) > 100:
            self._undo_stack.pop(0)

    def _handle_undo(self):
        if not self._undo_stack:
            self.window.statusbar.showMessage("Nothing to undo.", 2000)
            return
        snapshot = self._undo_stack.pop()
        self.state.restore_edit_state(snapshot)
        self._refresh_formants_from_state()
        self._drag_edit_active = False
        self._mark_current_entry_dirty()
        self.window.statusbar.showMessage("Undid last edit.", 2000)

    def _default_export_path(self, suffix):
        if self.current_entry_index < 0 or self.current_entry_index >= len(self.batch_entries):
            return ""
        audio_path = Path(self.batch_entries[self.current_entry_index].filepath).resolve()
        return str(audio_path.with_name(f"{audio_path.stem}{suffix}"))

    def _choose_save_file(self, title, default_path, file_filter):
        dialog = QFileDialog(self.window, title, default_path)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        self._configure_file_dialog(dialog, default_path)
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
        self._configure_file_dialog(dialog, start_dir)
        if dialog.exec() != QFileDialog.Accepted:
            return ""
        selected = dialog.selectedFiles()
        return selected[0] if selected else ""

    def _configure_file_dialog(self, dialog, reference_path=""):
        sidebar_paths = []

        def add_path(path_value):
            if not path_value:
                return
            path = Path(path_value).expanduser()
            candidate = path if path.is_dir() else path.parent
            if not candidate.exists():
                return
            resolved = str(candidate.resolve())
            if resolved not in sidebar_paths:
                sidebar_paths.append(resolved)

        home_dir = Path.home()
        add_path(home_dir)
        for location in (
            QStandardPaths.DesktopLocation,
            QStandardPaths.DocumentsLocation,
            QStandardPaths.DownloadLocation,
            QStandardPaths.MoviesLocation,
            QStandardPaths.MusicLocation,
        ):
            for found_path in QStandardPaths.standardLocations(location):
                add_path(found_path)

        add_path(Path(__file__).resolve().parent)
        add_path(reference_path)
        if 0 <= self.current_entry_index < len(self.batch_entries):
            add_path(self.batch_entries[self.current_entry_index].filepath)

        if sidebar_paths:
            dialog.setSidebarUrls([QUrl.fromLocalFile(path) for path in sidebar_paths])
            dialog.setHistory(sidebar_paths)

    def _current_pitch_params(self):
        return {
            "pitch_floor": float(self.state.pitch_floor),
            "pitch_ceiling": float(self.state.pitch_ceiling),
            "time_step": float(self.state.time_step),
            "voicing_threshold": float(self.state.voicing_threshold),
            "silence_threshold": float(self.state.silence_threshold),
            "octave_cost": float(self.state.octave_cost),
            "octave_jump_cost": float(self.state.octave_jump_cost),
            "voiced_unvoiced_cost": float(self.state.voiced_unvoiced_cost),
        }

    def _compute_entry_acoustic_row(self, entry: BatchAudioEntry):
        if entry.state_snapshot is None:
            return compute_feature_row_with_pitch_overrides(entry.filepath, entry.params)
        row = compute_feature_row_with_pitch_overrides(
            entry.filepath,
            entry.params,
            entry.state_snapshot["timestamps"],
            entry.state_snapshot["pitch_values"],
            entry.state_snapshot["segment_labels"],
        )
        entry.acoustic_row = row
        return row

    def _entry_export_payload(self, entry: BatchAudioEntry):
        if entry.state_snapshot is None:
            return {
                "audio_path": entry.filepath,
                "pitch_params": dict(entry.params),
                "timestamps": np.array([], dtype=float),
                "pitch_values": np.array([], dtype=float),
                "segment_labels": np.array([], dtype=int),
            }
        snapshot = entry.state_snapshot
        return {
            "audio_path": entry.filepath,
            "pitch_params": dict(entry.params),
            "timestamps": np.array(snapshot["timestamps"], copy=True),
            "pitch_values": np.array(snapshot["pitch_values"], copy=True),
            "segment_labels": np.array(snapshot["segment_labels"], copy=True),
        }

    def _current_export_payload(self):
        return {
            "audio_path": self.state.audio_path,
            "pitch_params": self._current_pitch_params(),
            "timestamps": np.array(self.state.timestamps, copy=True),
            "pitch_values": np.array(self.state.pitch_values, copy=True),
            "segment_labels": np.array(self.state.segment_labels, copy=True),
        }

    def _start_export_task(self, task, start_message):
        if self._export_in_progress:
            self.window.statusbar.showMessage("An export is already running. Please wait.", 3000)
            return
        self._export_in_progress = True
        self.window.statusbar.showMessage(start_message)
        self.request_export.emit(task)

    def _handle_export_csv(self):
        if self.current_entry_index < 0:
            return
        filepath = self._choose_save_file("Export CSV", self._default_export_path("_pitch.csv"), "CSV Files (*.csv)")
        if filepath:
            payload = self._current_export_payload()
            self._start_export_task(
                {
                    "type": "pitch_csv",
                    "filepath": filepath,
                    **payload,
                },
                "Exporting pitch CSV in background...",
            )

    def _handle_export_praat(self):
        if self.current_entry_index < 0:
            return
        filepath = self._choose_save_file("Export Praat Pitch", self._default_export_path(".Pitch"), "Pitch Files (*.Pitch)")
        if filepath:
            payload = self._current_export_payload()
            self._start_export_task(
                {
                    "type": "praat_pitch",
                    "filepath": filepath,
                    "timestamps": payload["timestamps"],
                    "pitch_values": payload["pitch_values"],
                },
                "Exporting Praat Pitch in background...",
            )

    def _handle_export_acoustic_csv(self):
        if self.current_entry_index < 0:
            self.window.statusbar.showMessage("No audio loaded.", 3000)
            return
        entry = self.batch_entries[self.current_entry_index]
        filepath = self._choose_save_file("Export Acoustic Features CSV", self._default_export_path("_acoustic_features.csv"), "CSV Files (*.csv)")
        if filepath:
            self._save_current_entry_state()
            payload = self._entry_export_payload(entry)
            self._start_export_task(
                {
                    "type": "acoustic_csv",
                    "filepath": filepath,
                    **payload,
                },
                "Exporting acoustic features in background...",
            )

    def _handle_batch_export_acoustic_csv(self):
        if not self.batch_entries:
            self.window.statusbar.showMessage("No audio files imported.", 3000)
            return
        default_parent = str(Path(self.batch_entries[0].filepath).resolve().parent / "batch_acoustic_features.csv")
        filepath = self._choose_save_file("Export Batch Acoustic Features CSV", default_parent, "CSV Files (*.csv)")
        if not filepath:
            return
        self._save_current_entry_state()
        entries = [self._entry_export_payload(entry) for entry in self.batch_entries]
        self._start_export_task(
            {
                "type": "batch_acoustic_csv",
                "filepath": filepath,
                "entries": entries,
            },
            f"Exporting batch acoustic features for {len(entries)} files in background...",
        )

    def _handle_export_all(self):
        if self.current_entry_index < 0:
            self.window.statusbar.showMessage("No audio loaded.", 3000)
            return
        audio_path = Path(self.batch_entries[self.current_entry_index].filepath).resolve()
        output_dir = self._choose_directory("Export All", str(audio_path.parent))
        if not output_dir:
            return
        stem = audio_path.stem
        base_dir = Path(output_dir)
        payload = self._current_export_payload()
        self._start_export_task(
            {
                "type": "export_all",
                "output_dir": str(base_dir),
                "pitch_csv_path": str(base_dir / f"{stem}_pitch.csv"),
                "praat_path": str(base_dir / f"{stem}.Pitch"),
                "acoustic_csv_path": str(base_dir / f"{stem}_acoustic_features.csv"),
                **payload,
            },
            "Exporting all files in background...",
        )

    def _handle_volume_changed(self, value):
        self._playback_volume = max(0.0, min(1.0, value / 100.0))
        if self._audio_sink is not None:
            self._audio_sink.setVolume(self._playback_volume)

    def _refresh_audio_output_devices(self):
        self._audio_output_devices = list(QMediaDevices.audioOutputs())
        default_device = QMediaDevices.defaultAudioOutput()
        default_id = bytes(default_device.id()) if hasattr(default_device, "id") else b""
        selected_index = 0
        device_names = []
        for idx, device in enumerate(self._audio_output_devices):
            device_names.append(device.description())
            device_id = bytes(device.id()) if hasattr(device, "id") else b""
            if default_id and device_id == default_id:
                selected_index = idx
        if not device_names:
            device_names = ["System Default"]
            self._selected_audio_output_index = 0
        else:
            self._selected_audio_output_index = min(self._selected_audio_output_index, len(device_names) - 1)
            if self._selected_audio_output_index == 0:
                self._selected_audio_output_index = selected_index
        self.window.control_panel.set_audio_output_devices(device_names, self._selected_audio_output_index)

    def _handle_audio_output_device_changed(self, index):
        if not self._audio_output_devices:
            self._selected_audio_output_index = 0
            return
        self._selected_audio_output_index = max(0, min(int(index), len(self._audio_output_devices) - 1))
        if self._audio_sink is not None:
            self.window.statusbar.showMessage(
                f"Playback device set to {self._audio_output_devices[self._selected_audio_output_index].description()}",
                3000,
            )

    def _mark_current_entry_dirty(self):
        if self._suppress_dirty_tracking:
            return
        if self.current_entry_index < 0 or self.current_entry_index >= len(self.batch_entries):
            return
        entry = self.batch_entries[self.current_entry_index]
        if not entry.dirty:
            entry.dirty = True
            self.window.update_audio_file_entry(self.current_entry_index, entry.filepath, True)

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
        self._play_clip(clip, sr, f"Playing selection {start_time:.3f}s - {end_time:.3f}s")

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
        self._play_clip(clip, sr, f"Playing pitch track {start_time:.3f}s - {end_time:.3f}s")

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
        freq_samples = np.interp(sample_times, frame_times, np.where(voiced_frame_mask, frame_freqs, 0.0), left=0.0, right=0.0)
        amp_samples = np.interp(sample_times, frame_times, np.where(voiced_frame_mask, 1.0, 0.0), left=0.0, right=0.0)
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
        return waveform.astype(np.float32) * amp_samples

    def _play_clip(self, clip, sr, status_message):
        clip = np.asarray(clip, dtype=np.float32)
        if clip.size == 0:
            self.window.statusbar.showMessage("Selected region is empty.", 3000)
            return

        if self._audio_sink is not None:
            self._audio_sink.stop()
            self._audio_sink.deleteLater()
            self._audio_sink = None
        if self._audio_buffer is not None:
            self._audio_buffer.close()
            self._audio_buffer.deleteLater()
            self._audio_buffer = None

        if self._audio_output_devices and 0 <= self._selected_audio_output_index < len(self._audio_output_devices):
            audio_device = self._audio_output_devices[self._selected_audio_output_index]
        else:
            audio_device = QMediaDevices.defaultAudioOutput()
        preferred = audio_device.preferredFormat()
        out_sr = preferred.sampleRate() if preferred.sampleRate() > 0 else int(sr)
        out_channels = preferred.channelCount() if preferred.channelCount() > 0 else 2
        out_format = preferred.sampleFormat()

        if out_sr != int(sr) and out_sr > 0 and clip.size > 1:
            target_count = max(1, int(math.ceil(len(clip) * out_sr / float(sr))))
            source_x = np.linspace(0.0, 1.0, num=len(clip), endpoint=False)
            target_x = np.linspace(0.0, 1.0, num=target_count, endpoint=False)
            clip = np.interp(target_x, source_x, clip).astype(np.float32)

        clip = np.clip(clip, -1.0, 1.0)
        if out_channels > 1:
            samples = np.repeat(clip[:, None], out_channels, axis=1)
        else:
            samples = clip[:, None]

        if out_format == QAudioFormat.SampleFormat.UInt8:
            pcm_bytes = np.asarray((samples * 127.5) + 127.5, dtype=np.uint8).tobytes()
        elif out_format == QAudioFormat.SampleFormat.Int16:
            pcm_bytes = np.asarray(samples * 32767.0, dtype=np.int16).tobytes()
        elif out_format == QAudioFormat.SampleFormat.Int32:
            pcm_bytes = np.asarray(samples * 2147483647.0, dtype=np.int32).tobytes()
        else:
            pcm_bytes = np.asarray(samples, dtype=np.float32).tobytes()

        byte_array = QByteArray(pcm_bytes)
        buffer = QBuffer(self.window)
        buffer.setData(byte_array)
        buffer.open(QIODevice.ReadOnly)

        sink = QAudioSink(audio_device, preferred, self.window)
        sink.setVolume(self._playback_volume)
        sink.start(buffer)

        self._audio_buffer = buffer
        self._audio_sink = sink
        self.window.statusbar.showMessage(
            f"{status_message} [{out_sr} Hz, {out_channels} ch]",
            3000,
        )

    def _on_error(self, err_msg):
        QMessageBox.critical(self.window, "Error", err_msg)
        self.window.statusbar.showMessage("Error occurred.")
        self._loading_entry_index = -1

    def _on_export_finished(self, message):
        self._export_in_progress = False
        self.window.statusbar.showMessage(message, 4000)

    def _on_export_error(self, err_msg):
        self._export_in_progress = False
        QMessageBox.critical(self.window, "Export Error", err_msg)
        self.window.statusbar.showMessage("Export failed.", 4000)

    def cleanup(self):
        if self._cleaned_up:
            return
        self._cleaned_up = True
        if self._audio_sink is not None:
            self._audio_sink.stop()
            self._audio_sink.deleteLater()
            self._audio_sink = None
        if self._audio_buffer is not None:
            self._audio_buffer.close()
            self._audio_buffer.deleteLater()
            self._audio_buffer = None
        self.thread.quit()
        self.thread.wait()
        self.thread.deleteLater()
        self.export_thread.quit()
        self.export_thread.wait()
        self.export_thread.deleteLater()


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
