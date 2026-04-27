from pathlib import Path
import math

from PySide6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout,
                               QVBoxLayout, QListWidget, QListWidgetItem,
                               QStatusBar, QLabel, QGroupBox, QSplitter,
                               QSplitterHandle)
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
    current_audio_index_changed = Signal(int)
    next_audio_requested = Signal()
    previous_audio_requested = Signal()

    def __init__(self):
        super().__init__()
        self.close_handler = None
        self.setWindowTitle("Pitch Annotator")
        self.resize(1700, 850)
        
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

        sidebar = QWidget()
        sidebar.setMinimumWidth(120)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(8)
        group_files = QGroupBox("Batch Audio Files")
        files_layout = QVBoxLayout(group_files)
        self.audio_list = QListWidget()
        self.audio_list.currentRowChanged.connect(self.current_audio_index_changed.emit)
        files_layout.addWidget(self.audio_list)
        self.lbl_batch_hint = QLabel("Import multiple audio files, then select one here or use Up/Down.")
        self.lbl_batch_hint.setWordWrap(True)
        files_layout.addWidget(self.lbl_batch_hint)
        sidebar_layout.addWidget(group_files)

        self.canvas = PitchCanvas()
        self.control_panel = ControlPanel()

        right_splitter = QSplitter(Qt.Horizontal)
        right_splitter.setHandleWidth(4)
        right_splitter.addWidget(self.canvas)
        right_splitter.addWidget(self.control_panel)
        right_splitter.setSizes([650, 400])
        right_splitter.setCollapsible(0, False)
        right_splitter.setCollapsible(1, False)
        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 2)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)
        splitter.addWidget(sidebar)
        splitter.addWidget(right_splitter)
        splitter.setSizes([350, 850])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)
        for handle in splitter.findChildren(QSplitterHandle):
            handle.setCursor(Qt.SplitHCursor)
        for handle in right_splitter.findChildren(QSplitterHandle):
            handle.setCursor(Qt.SplitHCursor)
        
        self._setup_menus()
        self._setup_statusbar()
        self._setup_shortcuts()

    def _setup_menus(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        
        action_open = file_menu.addAction("Import Audio Files...")
        action_open.triggered.connect(self.open_audio_requested.emit)
        
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
        self.lbl_current_file.setMinimumWidth(320)
        self.lbl_stats = QLabel("F0 Stats: N/A")
        self.lbl_voice = QLabel("Voice%: N/A")
        self.lbl_pitch_source = QLabel("Pitch source: Unknown")
        self.lbl_duration = QLabel("Total: 0.000s | Selection: 0.000s")
        self.statusbar.addWidget(self.lbl_current_file, 1)
        self.statusbar.addWidget(self.lbl_duration)
        self.statusbar.addPermanentWidget(self.lbl_stats)
        self.statusbar.addPermanentWidget(self.lbl_voice)
        self.statusbar.addPermanentWidget(self.lbl_pitch_source)

    def _setup_shortcuts(self):
        self.shortcut_play_selection = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.shortcut_play_selection.activated.connect(self.play_selection_requested.emit)
        self.shortcut_play_pitch_track = QShortcut(QKeySequence("Shift+Space"), self)
        self.shortcut_play_pitch_track.activated.connect(self.play_pitch_track_requested.emit)
        self.shortcut_undo = QShortcut(QKeySequence.Undo, self)
        self.shortcut_undo.activated.connect(self.undo_requested.emit)
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
        self.lbl_stats.setText(
            "F0 "
            f"20%: {p20:.1f}Hz ({p20_st:.1f}st) | "
            f"50%: {p50:.1f}Hz ({p50_st:.1f}st) | "
            f"80%: {p80:.1f}Hz ({p80_st:.1f}st)"
        )
        self.lbl_voice.setText(f"Voice%: {voice_percent:.1f}")

    def update_durations(self, total_duration, selection_duration):
        self.lbl_duration.setText(
            f"Total: {total_duration:.3f}s | Selection: {selection_duration:.3f}s"
        )

    def update_pitch_source(self, source_text):
        self.lbl_pitch_source.setText(f"Pitch source: {source_text}")

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
