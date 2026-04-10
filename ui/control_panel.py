from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                               QLabel, QSpinBox, QDoubleSpinBox, QSlider, QComboBox,
                               QPushButton, QGroupBox, QFormLayout)
from PySide6.QtCore import Signal, Qt

class ControlPanel(QWidget):
    # Signals
    parameters_changed = Signal(float, float, float, float, float, float, float, float, float)
    recompute_requested = Signal()
    region_toggled = Signal(bool)
    set_region_voiced = Signal()
    set_region_unvoiced = Signal()
    set_region_silence = Signal()
    apply_params_to_all_requested = Signal()
    volume_changed = Signal(int)
    audio_output_device_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(250)
        
        main_layout = QVBoxLayout(self)
        
        # Audio Parameters Group
        group_params = QGroupBox("Pitch Parameters")
        form_layout = QFormLayout()

        self.lbl_pitch_method = QLabel("Pitch method: Filtered AC")
        self.lbl_pitch_method.setWordWrap(True)
        form_layout.addRow(self.lbl_pitch_method)
        
        self.spin_floor = QSpinBox()
        self.spin_floor.setRange(20, 300)
        self.spin_floor.setValue(50)
        self.spin_floor.setSuffix(" Hz")
        
        self.spin_ceiling = QSpinBox()
        self.spin_ceiling.setRange(100, 1200)
        self.spin_ceiling.setValue(800)
        self.spin_ceiling.setSuffix(" Hz")
        
        self.spin_step = QDoubleSpinBox()
        self.spin_step.setRange(0.0, 0.1)
        self.spin_step.setSingleStep(0.005)
        self.spin_step.setDecimals(3)
        self.spin_step.setValue(0.0)
        self.spin_step.setSuffix(" s")

        self.spin_filtered_ac_attenuation = QDoubleSpinBox()
        self.spin_filtered_ac_attenuation.setRange(0.001, 1.0)
        self.spin_filtered_ac_attenuation.setSingleStep(0.01)
        self.spin_filtered_ac_attenuation.setDecimals(3)
        self.spin_filtered_ac_attenuation.setValue(0.03)

        self.spin_voicing_threshold = QDoubleSpinBox()
        self.spin_voicing_threshold.setRange(0.0, 1.0)
        self.spin_voicing_threshold.setSingleStep(0.01)
        self.spin_voicing_threshold.setDecimals(2)
        self.spin_voicing_threshold.setValue(0.50)

        self.spin_silence_threshold = QDoubleSpinBox()
        self.spin_silence_threshold.setRange(0.0, 1.0)
        self.spin_silence_threshold.setSingleStep(0.01)
        self.spin_silence_threshold.setDecimals(2)
        self.spin_silence_threshold.setValue(0.09)

        self.spin_octave_cost = QDoubleSpinBox()
        self.spin_octave_cost.setRange(0.0, 1.0)
        self.spin_octave_cost.setSingleStep(0.01)
        self.spin_octave_cost.setDecimals(3)
        self.spin_octave_cost.setValue(0.055)

        self.spin_octave_jump_cost = QDoubleSpinBox()
        self.spin_octave_jump_cost.setRange(0.0, 2.0)
        self.spin_octave_jump_cost.setSingleStep(0.01)
        self.spin_octave_jump_cost.setDecimals(2)
        self.spin_octave_jump_cost.setValue(0.35)

        self.spin_voiced_unvoiced_cost = QDoubleSpinBox()
        self.spin_voiced_unvoiced_cost.setRange(0.0, 1.0)
        self.spin_voiced_unvoiced_cost.setSingleStep(0.01)
        self.spin_voiced_unvoiced_cost.setDecimals(2)
        self.spin_voiced_unvoiced_cost.setValue(0.14)
        
        form_layout.addRow("Pitch Floor:", self.spin_floor)
        form_layout.addRow("Pitch Top:", self.spin_ceiling)
        form_layout.addRow("Time Step:", self.spin_step)
        form_layout.addRow("Attenuation at Top:", self.spin_filtered_ac_attenuation)
        form_layout.addRow("Voicing Threshold:", self.spin_voicing_threshold)
        form_layout.addRow("Silence Threshold:", self.spin_silence_threshold)
        form_layout.addRow("Octave Cost:", self.spin_octave_cost)
        form_layout.addRow("Octave Jump Cost:", self.spin_octave_jump_cost)
        form_layout.addRow("Voiced/Unvoiced Cost:", self.spin_voiced_unvoiced_cost)
        
        btn_recompute = QPushButton("Recompute Initial Pitch")
        self.btn_apply_all = QPushButton("Apply Current Params\nTo All Imported Files")
        self.btn_apply_all.setToolTip(
            "Copy the current pitch parameters to every imported audio file. "
            "This updates their defaults, but you should still review each file manually."
        )
        self.btn_apply_all.setMinimumHeight(52)
        form_layout.addRow(btn_recompute)
        form_layout.addRow(self.btn_apply_all)
        group_params.setLayout(form_layout)
        
        # Editing Tools Group
        group_tools = QGroupBox("Manual Editing Tools")
        vbox_tools = QVBoxLayout()
        
        self.btn_toggle_region = QPushButton("Show Region Box")
        self.btn_toggle_region.setCheckable(True)
        
        btn_set_voiced = QPushButton("Set Region to Voiced")
        btn_set_unvoiced = QPushButton("Set Region to Unvoiced")
        btn_set_silence = QPushButton("Set Region to Silence")
        
        vbox_tools.addWidget(QLabel("Instructions:\n- Alt+Left Click: Add Snapped Point\n- Shift+Left Click: Remove Point\n- Space: Play Selected Region\n- Shift+Space: Play Pitch Track"))
        vbox_tools.addWidget(self.btn_toggle_region)
        vbox_tools.addWidget(btn_set_voiced)
        vbox_tools.addWidget(btn_set_unvoiced)
        vbox_tools.addWidget(btn_set_silence)
        group_tools.setLayout(vbox_tools)

        group_playback = QGroupBox("Playback")
        playback_layout = QFormLayout()
        self.combo_audio_output = QComboBox()
        playback_layout.addRow("Output Device:", self.combo_audio_output)
        self.slider_volume = QSlider(Qt.Horizontal)
        self.slider_volume.setRange(0, 100)
        self.slider_volume.setValue(100)
        self.lbl_volume_value = QLabel("100%")
        volume_row = QHBoxLayout()
        volume_row.addWidget(self.slider_volume)
        volume_row.addWidget(self.lbl_volume_value)
        playback_layout.addRow("Volume:", volume_row)
        group_playback.setLayout(playback_layout)
        
        main_layout.addWidget(group_params)
        main_layout.addWidget(group_tools)
        main_layout.addWidget(group_playback)
        main_layout.addStretch()

        # Connect internal signals
        btn_recompute.clicked.connect(self._on_recompute)
        self.btn_apply_all.clicked.connect(self.apply_params_to_all_requested.emit)
        self.btn_toggle_region.toggled.connect(self.region_toggled.emit)
        btn_set_voiced.clicked.connect(self.set_region_voiced.emit)
        btn_set_unvoiced.clicked.connect(self.set_region_unvoiced.emit)
        btn_set_silence.clicked.connect(self.set_region_silence.emit)
        self.slider_volume.valueChanged.connect(self._on_volume_changed)
        self.combo_audio_output.currentIndexChanged.connect(self.audio_output_device_changed.emit)
        
    def _on_recompute(self):
        f = float(self.spin_floor.value())
        c = float(self.spin_ceiling.value())
        s = float(self.spin_step.value())
        a = float(self.spin_filtered_ac_attenuation.value())
        vt = float(self.spin_voicing_threshold.value())
        st = float(self.spin_silence_threshold.value())
        oc = float(self.spin_octave_cost.value())
        ojc = float(self.spin_octave_jump_cost.value())
        vuc = float(self.spin_voiced_unvoiced_cost.value())
        self.parameters_changed.emit(f, c, s, a, vt, st, oc, ojc, vuc)
        self.recompute_requested.emit()

    def _on_volume_changed(self, value):
        self.lbl_volume_value.setText(f"{value}%")
        self.volume_changed.emit(int(value))

    def set_audio_output_devices(self, device_names, current_index=0):
        self.combo_audio_output.blockSignals(True)
        self.combo_audio_output.clear()
        self.combo_audio_output.addItems(device_names)
        if 0 <= current_index < len(device_names):
            self.combo_audio_output.setCurrentIndex(current_index)
        self.combo_audio_output.blockSignals(False)
