from pathlib import Path
import math

from PySide6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout,
                               QVBoxLayout, QListWidget, QListWidgetItem,
                               QStatusBar, QLabel, QGroupBox, QSplitter,
                               QSplitterHandle, QPushButton, QApplication)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtGui import QKeySequence, QShortcut
from ui.canvas import PitchCanvas
from ui.control_panel import ControlPanel


def hz_to_semitone(freq_hz: float) -> float:
    if freq_hz <= 0:
        return float("nan")
    return 12.0 * math.log2(freq_hz / 27.5)


class MainWindow(QMainWindow):
    open_audio_requested = Signal()
    import_pitch_csv_requested = Signal()
    batch_export_acoustic_csv_requested = Signal()
    batch_export_pitch_csv_requested = Signal()
    batch_export_spectrograms_requested = Signal()
    batch_export_all_requested = Signal()
    export_csv_requested = Signal()
    export_spectrogram_requested = Signal()
    export_praat_requested = Signal()
    export_acoustic_csv_requested = Signal()
    export_all_requested = Signal()
    play_selection_requested = Signal()
    play_pitch_track_requested = Signal()
    undo_requested = Signal()
    set_region_voiced_requested = Signal()
    set_region_unvoiced_requested = Signal()
    set_region_silence_requested = Signal()
    current_audio_index_changed = Signal(int)
    next_audio_requested = Signal()
    previous_audio_requested = Signal()
    clear_audio_list_requested = Signal()

    def __init__(self):
        super().__init__()
        self.close_handler = None
        self.setWindowTitle("Pitch Annotator")
        self.setMinimumSize(900, 600)

        screen = QApplication.primaryScreen()
        if screen is not None:
            geom = screen.availableGeometry()
            w = min(int(geom.width() * 0.92), 1440)
            h = min(int(geom.height() * 0.88), 850)
        else:
            w, h = 1400, 800
        self.resize(w, h)
        
        # Apply simple dark theme
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #2b2b2b;
                color: #e0e0e0;
            }
            QPushButton {
                background-color: #3c3f41;
                border: 1px solid #555555;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #4c5052;
            }
            QGroupBox {
                border: 1px solid #555555;
                margin-top: 1ex;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                padding: 0 3px;
            }
            QSplitter::handle {
                background-color: #555555;
                width: 3px;
                margin: 1px 0;
            }
            QSplitter::handle:hover {
                background-color: #888888;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QHBoxLayout(central_widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(0)

        sidebar = QWidget()
        sidebar.setMinimumWidth(160)
        sidebar.setMaximumWidth(260)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 6, 0)
        sidebar_layout.setSpacing(6)

        group_files = QGroupBox("Batch Files")
        files_layout = QVBoxLayout(group_files)
        self.audio_list = QListWidget()
        self.audio_list.currentRowChanged.connect(self.current_audio_index_changed.emit)
        files_layout.addWidget(self.audio_list)
        self.lbl_batch_hint = QLabel("Up/Down to switch file")
        self.lbl_batch_hint.setWordWrap(True)
        self.lbl_batch_hint.setToolTip("Import multiple audio files, then select one here or use Up/Down keys.")
        files_layout.addWidget(self.lbl_batch_hint)

        self.btn_clear_list = QPushButton("Clear File List")
        self.btn_clear_list.clicked.connect(self.clear_audio_list_requested.emit)
        files_layout.addWidget(self.btn_clear_list)

        sidebar_layout.addWidget(group_files)

        self.canvas = PitchCanvas()
        self.control_panel = ControlPanel()

        right_splitter = QSplitter(Qt.Horizontal)
        right_splitter.setHandleWidth(4)
        self.right_splitter = right_splitter
        right_splitter.addWidget(self.canvas)
        right_splitter.addWidget(self.control_panel)
        right_splitter.setCollapsible(0, False)
        right_splitter.setCollapsible(1, False)
        right_splitter.setStretchFactor(0, 4)
        right_splitter.setStretchFactor(1, 1)

        sidebar_width = 190
        control_width = 300
        right_space = max(600, w - sidebar_width)
        right_splitter.setSizes([max(500, right_space - control_width), control_width])

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setHandleWidth(4)
        self.splitter.addWidget(sidebar)
        self.splitter.addWidget(right_splitter)
        self.splitter.setSizes([sidebar_width, right_space])
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        layout.addWidget(self.splitter)
        
        self._setup_menus()
        self._setup_statusbar()
        self._setup_shortcuts()

        for handle in self.splitter.findChildren(QSplitterHandle):
            handle.setCursor(Qt.SplitHCursor)
        for handle in self.right_splitter.findChildren(QSplitterHandle):
            handle.setCursor(Qt.SplitHCursor)

    def _setup_menus(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        
        action_open = file_menu.addAction("Import Audio Files...")
        action_open.triggered.connect(self.open_audio_requested.emit)

        action_import_pitch_csv = file_menu.addAction("Import Pitch CSVs...")
        action_import_pitch_csv.triggered.connect(self.import_pitch_csv_requested.emit)
        
        file_menu.addSeparator()
        
        action_export_csv = file_menu.addAction("Export CSV...")
        action_export_csv.triggered.connect(self.export_csv_requested.emit)

        action_export_spectrogram = file_menu.addAction("Export Spectrogram Plot...")
        action_export_spectrogram.triggered.connect(self.export_spectrogram_requested.emit)

        action_export_batch_pitch = file_menu.addAction("Export Batch Pitch CSVs...")
        action_export_batch_pitch.triggered.connect(self.batch_export_pitch_csv_requested.emit)

        action_export_batch_spectrograms = file_menu.addAction("Export Batch Spectrogram Plots...")
        action_export_batch_spectrograms.triggered.connect(self.batch_export_spectrograms_requested.emit)
        
        action_export_praat = file_menu.addAction("Export Praat .Pitch...")
        action_export_praat.triggered.connect(self.export_praat_requested.emit)

        action_export_acoustic = file_menu.addAction("Export Acoustic Features CSV...")
        action_export_acoustic.triggered.connect(self.export_acoustic_csv_requested.emit)

        action_export_batch_acoustic = file_menu.addAction("Export Batch Acoustic Features CSV...")
        action_export_batch_acoustic.triggered.connect(self.batch_export_acoustic_csv_requested.emit)

        action_export_batch_all = file_menu.addAction("Export Batch All...")
        action_export_batch_all.triggered.connect(self.batch_export_all_requested.emit)

        action_export_all = file_menu.addAction("Export All...")
        action_export_all.triggered.connect(self.export_all_requested.emit)

    def _setup_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.lbl_current_file = QLabel("Current file: None")
        self.lbl_current_file.setMinimumWidth(220)
        self.lbl_duration = QLabel("Total: 0.000s | Selection: 0.000s")
        self.lbl_selected_point = QLabel("Point: N/A")
        self.lbl_selected_point.setMinimumWidth(150)
        self.lbl_selected_point.setToolTip("Selected pitch point: N/A")
        self.lbl_voice = QLabel("Voice%: N/A")
        self.lbl_voice.setToolTip("F0 Stats: N/A | Pitch source: Unknown")
        self.lbl_pitch_source = QLabel("Pitch source: Unknown")
        self.lbl_pitch_source.setMinimumWidth(170)
        self.lbl_pitch_source.setToolTip("Pitch source: Unknown")
        self.statusbar.addWidget(self.lbl_current_file, 1)
        self.statusbar.addWidget(self.lbl_duration)
        self.statusbar.addPermanentWidget(self.lbl_selected_point)
        self.statusbar.addPermanentWidget(self.lbl_voice)
        self.statusbar.addPermanentWidget(self.lbl_pitch_source)

    def _setup_shortcuts(self):
        self.shortcut_play_selection = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.shortcut_play_selection.activated.connect(self.play_selection_requested.emit)
        self.shortcut_play_pitch_track = QShortcut(QKeySequence("Shift+Space"), self)
        self.shortcut_play_pitch_track.activated.connect(self.play_pitch_track_requested.emit)
        self.shortcut_undo = QShortcut(QKeySequence.Undo, self)
        self.shortcut_undo.activated.connect(self.undo_requested.emit)
        self.shortcut_set_region_voiced = QShortcut(QKeySequence("1"), self)
        self.shortcut_set_region_voiced.activated.connect(self.set_region_voiced_requested.emit)
        self.shortcut_set_region_unvoiced = QShortcut(QKeySequence("2"), self)
        self.shortcut_set_region_unvoiced.activated.connect(self.set_region_unvoiced_requested.emit)
        self.shortcut_set_region_silence = QShortcut(QKeySequence("3"), self)
        self.shortcut_set_region_silence.activated.connect(self.set_region_silence_requested.emit)
        self.shortcut_next_audio = QShortcut(QKeySequence(Qt.Key_Down), self)
        self.shortcut_next_audio.activated.connect(self.next_audio_requested.emit)
        self.shortcut_previous_audio = QShortcut(QKeySequence(Qt.Key_Up), self)
        self.shortcut_previous_audio.activated.connect(self.previous_audio_requested.emit)
        self.shortcut_export_all = QShortcut(QKeySequence("Ctrl+Shift+E"), self)
        self.shortcut_export_all.activated.connect(self.export_all_requested.emit)
        self.shortcut_batch_export_all = QShortcut(QKeySequence("Ctrl+Alt+Shift+E"), self)
        self.shortcut_batch_export_all.activated.connect(self.batch_export_all_requested.emit)

    def update_stats(self, p20, p50, p80, voice_percent):
        p20_st = hz_to_semitone(float(p20))
        p50_st = hz_to_semitone(float(p50))
        p80_st = hz_to_semitone(float(p80))
        if not math.isfinite(float(p20)) or float(p20) <= 0:
            self._last_stats_text = "F0 Stats: N/A"
            self.lbl_voice.setText(f"Voice%: {voice_percent:.1f} | F0: N/A")
        else:
            self._last_stats_text = (
                f"F0 20%: {p20:.1f}Hz ({p20_st:.1f}st) | "
                f"50%: {p50:.1f}Hz ({p50_st:.1f}st) | "
                f"80%: {p80:.1f}Hz ({p80_st:.1f}st)"
            )
            self.lbl_voice.setText(
                f"Voice%: {voice_percent:.1f} | "
                f"F0 20/50/80: {p20:.1f}/{p50:.1f}/{p80:.1f}Hz"
            )
        source = getattr(self, "_last_pitch_source", "")
        tip = self._last_stats_text
        if source:
            tip += f" | Pitch source: {source}"
        self.lbl_voice.setToolTip(tip)

    def update_durations(self, total_duration, selection_duration):
        self.lbl_duration.setText(
            f"Total: {total_duration:.3f}s | Selection: {selection_duration:.3f}s"
        )

    def update_selected_pitch_point(self, time_value, pitch_value):
        if time_value is None or pitch_value is None:
            self.lbl_selected_point.setText("Point: N/A")
            self.lbl_selected_point.setToolTip("Selected pitch point: N/A")
            return
        pitch = float(pitch_value)
        time_pos = float(time_value)
        semitone = hz_to_semitone(pitch)
        text = f"Point: {pitch:.1f}Hz @ {time_pos:.3f}s"
        self.lbl_selected_point.setText(text)
        self.lbl_selected_point.setToolTip(
            f"Selected pitch point: {pitch:.3f} Hz ({semitone:.3f} st) at {time_pos:.6f} s"
        )

    def update_pitch_source(self, source_text):
        source_text = source_text or "Unknown"
        self._last_pitch_source = source_text
        display_text = f"Pitch source: {source_text}"
        metrics = QFontMetrics(self.lbl_pitch_source.font())
        elided = metrics.elidedText(display_text, Qt.ElideMiddle, max(170, self.lbl_pitch_source.width() - 8))
        self.lbl_pitch_source.setText(elided)
        self.lbl_pitch_source.setToolTip(display_text)
        stats = getattr(self, "_last_stats_text", "")
        tip = stats if stats else ""
        if tip:
            tip += " | "
        tip += display_text
        self.lbl_voice.setToolTip(tip)

    def update_current_file(self, filepath):
        if not filepath:
            self.lbl_current_file.setText("Current file: None")
            return
        file_name = Path(filepath).name
        metrics = QFontMetrics(self.lbl_current_file.font())
        elided = metrics.elidedText(file_name, Qt.ElideMiddle, max(220, self.lbl_current_file.width() - 16))
        self.lbl_current_file.setText(f"Current file: {elided}")
        self.lbl_current_file.setToolTip(filepath)

    def set_audio_files(self, filepaths):
        self.audio_list.blockSignals(True)
        self.audio_list.clear()
        for filepath in filepaths:
            item = QListWidgetItem(Path(filepath).name)
            item.setToolTip(filepath)
            self.audio_list.addItem(item)
        self.audio_list.blockSignals(False)
        if not filepaths:
            self.update_current_file("")

    def set_current_audio_index(self, index):
        if index < 0 or index >= self.audio_list.count():
            return
        self.audio_list.blockSignals(True)
        self.audio_list.setCurrentRow(index)
        self.audio_list.blockSignals(False)

    def update_audio_file_entry(self, index, filepath, dirty):
        if index < 0 or index >= self.audio_list.count():
            return
        label = Path(filepath).name
        if dirty:
            label = f"* {label}"
        item = self.audio_list.item(index)
        item.setText(label)
        item.setToolTip(filepath)

    def closeEvent(self, event):
        if self.close_handler is not None and not self.close_handler():
            event.ignore()
            return
        super().closeEvent(event)
        QApplication.quit()
