from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)


class BatchImportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Import Preprocessing Parameters")
        self.resize(420, 260)

        layout = QVBoxLayout(self)
        hint = QLabel(
            "Choose the initial preprocessing / pitch parameters that will be applied to all imported audio files.\n"
            "Defaults follow Praat's default settings."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        form = QFormLayout()

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
        self.spin_step.setDecimals(3)
        self.spin_step.setSingleStep(0.005)
        self.spin_step.setValue(0.0)
        self.spin_step.setSuffix(" s")

        self.spin_filtered_ac_attenuation = QDoubleSpinBox()
        self.spin_filtered_ac_attenuation.setRange(0.001, 1.0)
        self.spin_filtered_ac_attenuation.setDecimals(3)
        self.spin_filtered_ac_attenuation.setSingleStep(0.01)
        self.spin_filtered_ac_attenuation.setValue(0.03)

        self.spin_voicing_threshold = QDoubleSpinBox()
        self.spin_voicing_threshold.setRange(0.0, 1.0)
        self.spin_voicing_threshold.setDecimals(2)
        self.spin_voicing_threshold.setSingleStep(0.01)
        self.spin_voicing_threshold.setValue(0.50)

        self.spin_silence_threshold = QDoubleSpinBox()
        self.spin_silence_threshold.setRange(0.0, 1.0)
        self.spin_silence_threshold.setDecimals(2)
        self.spin_silence_threshold.setSingleStep(0.01)
        self.spin_silence_threshold.setValue(0.09)

        self.spin_octave_cost = QDoubleSpinBox()
        self.spin_octave_cost.setRange(0.0, 1.0)
        self.spin_octave_cost.setDecimals(3)
        self.spin_octave_cost.setSingleStep(0.01)
        self.spin_octave_cost.setValue(0.055)

        self.spin_octave_jump_cost = QDoubleSpinBox()
        self.spin_octave_jump_cost.setRange(0.0, 2.0)
        self.spin_octave_jump_cost.setDecimals(2)
        self.spin_octave_jump_cost.setSingleStep(0.01)
        self.spin_octave_jump_cost.setValue(0.35)

        self.spin_voiced_unvoiced_cost = QDoubleSpinBox()
        self.spin_voiced_unvoiced_cost.setRange(0.0, 1.0)
        self.spin_voiced_unvoiced_cost.setDecimals(2)
        self.spin_voiced_unvoiced_cost.setSingleStep(0.01)
        self.spin_voiced_unvoiced_cost.setValue(0.14)

        form.addRow("Pitch Floor:", self.spin_floor)
        form.addRow("Pitch Top:", self.spin_ceiling)
        form.addRow("Time Step:", self.spin_step)
        form.addRow("Attenuation at Top:", self.spin_filtered_ac_attenuation)
        form.addRow("Voicing Threshold:", self.spin_voicing_threshold)
        form.addRow("Silence Threshold:", self.spin_silence_threshold)
        form.addRow("Octave Cost:", self.spin_octave_cost)
        form.addRow("Octave Jump Cost:", self.spin_octave_jump_cost)
        form.addRow("Voiced/Unvoiced Cost:", self.spin_voiced_unvoiced_cost)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_params(self):
        return {
            "pitch_floor": float(self.spin_floor.value()),
            "pitch_ceiling": float(self.spin_ceiling.value()),
            "time_step": float(self.spin_step.value()),
            "filtered_ac_attenuation_at_top": float(self.spin_filtered_ac_attenuation.value()),
            "voicing_threshold": float(self.spin_voicing_threshold.value()),
            "silence_threshold": float(self.spin_silence_threshold.value()),
            "octave_cost": float(self.spin_octave_cost.value()),
            "octave_jump_cost": float(self.spin_octave_jump_cost.value()),
            "voiced_unvoiced_cost": float(self.spin_voiced_unvoiced_cost.value()),
        }
