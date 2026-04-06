from PySide6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, 
                               QFileDialog, QStatusBar, QMessageBox, QLabel)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from ui.canvas import PitchCanvas
from ui.control_panel import ControlPanel

class MainWindow(QMainWindow):
    open_audio_requested = Signal()
    export_csv_requested = Signal()
    export_praat_requested = Signal()
    export_acoustic_csv_requested = Signal()
    export_all_requested = Signal()
    play_selection_requested = Signal()
    play_pitch_track_requested = Signal()
    undo_requested = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pitch Annotator")
        self.resize(1200, 800)
        
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
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QHBoxLayout(central_widget)
        
        self.canvas = PitchCanvas()
        self.control_panel = ControlPanel()
        
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.control_panel)
        
        self._setup_menus()
        self._setup_statusbar()
        self._setup_shortcuts()

    def _setup_menus(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        
        action_open = file_menu.addAction("Open Audio...")
        action_open.triggered.connect(self.open_audio_requested.emit)
        
        file_menu.addSeparator()
        
        action_export_csv = file_menu.addAction("Export CSV...")
        action_export_csv.triggered.connect(self.export_csv_requested.emit)
        
        action_export_praat = file_menu.addAction("Export Praat .Pitch...")
        action_export_praat.triggered.connect(self.export_praat_requested.emit)

        action_export_acoustic = file_menu.addAction("Export Acoustic Features CSV...")
        action_export_acoustic.triggered.connect(self.export_acoustic_csv_requested.emit)

        action_export_all = file_menu.addAction("Export All...")
        action_export_all.triggered.connect(self.export_all_requested.emit)

    def _setup_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.lbl_stats = QLabel("F0 Stats: N/A")
        self.lbl_voice = QLabel("Voice%: N/A")
        self.lbl_duration = QLabel("Total: 0.000s | Selection: 0.000s")
        self.statusbar.addWidget(self.lbl_duration)
        self.statusbar.addPermanentWidget(self.lbl_stats)
        self.statusbar.addPermanentWidget(self.lbl_voice)

    def _setup_shortcuts(self):
        self.shortcut_play_selection = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.shortcut_play_selection.activated.connect(self.play_selection_requested.emit)
        self.shortcut_play_pitch_track = QShortcut(QKeySequence("Shift+Space"), self)
        self.shortcut_play_pitch_track.activated.connect(self.play_pitch_track_requested.emit)
        self.shortcut_undo = QShortcut(QKeySequence.Undo, self)
        self.shortcut_undo.activated.connect(self.undo_requested.emit)
        self.shortcut_export_all = QShortcut(QKeySequence("Ctrl+Shift+E"), self)
        self.shortcut_export_all.activated.connect(self.export_all_requested.emit)

    def update_stats(self, p20, p50, p80, voice_percent):
        self.lbl_stats.setText(f"F0 20%: {p20:.1f}Hz | 50%: {p50:.1f}Hz | 80%: {p80:.1f}Hz")
        self.lbl_voice.setText(f"Voice%: {voice_percent:.1f}")

    def update_durations(self, total_duration, selection_duration):
        self.lbl_duration.setText(
            f"Total: {total_duration:.3f}s | Selection: {selection_duration:.3f}s"
        )
