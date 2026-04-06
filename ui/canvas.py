import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QWidget as QtWidget, QMenu, QGraphicsRectItem
from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QCursor, QColor, QPen, QBrush

class SelectionViewBox(pg.ViewBox):
    def __init__(self, selection_callback, point_drag_callback, zoom_anchor_callback, *args, **kwargs):
        super().__init__(*args, enableMenu=False, **kwargs)
        self.selection_callback = selection_callback
        self.point_drag_callback = point_drag_callback
        self.zoom_anchor_callback = zoom_anchor_callback
        self.setMouseMode(pg.ViewBox.PanMode)

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == Qt.LeftButton and ev.modifiers() == Qt.NoModifier:
            if self.point_drag_callback(ev):
                ev.accept()
                return
            ev.accept()
            start_point = self.mapSceneToView(ev.buttonDownScenePos())
            current_point = self.mapSceneToView(ev.scenePos())
            start_x = float(start_point.x())
            current_x = float(current_point.x())
            self.selection_callback(start_x, current_x, ev.isFinish())
            return
        super().mouseDragEvent(ev, axis=axis)

    def wheelEvent(self, ev, axis=None):
        ev.accept()
        delta = ev.delta()
        scale = 0.85 if delta > 0 else 1.18
        x_range, y_range = self.viewRange()
        mouse_view = self.mapSceneToView(ev.scenePos())
        anchor_x = float(mouse_view.x())
        anchor_y = float(self.zoom_anchor_callback(anchor_x))

        def scaled_range(current_min, current_max, anchor, factor):
            return anchor + (current_min - anchor) * factor, anchor + (current_max - anchor) * factor

        new_x_min, new_x_max = scaled_range(x_range[0], x_range[1], anchor_x, scale)
        new_y_min, new_y_max = scaled_range(y_range[0], y_range[1], anchor_y, scale)
        self.setXRange(new_x_min, new_x_max, padding=0)
        self.setYRange(new_y_min, new_y_max, padding=0)

class PitchCanvas(QWidget):
    # Signals to communicate intent to main Controller
    add_point_requested = Signal(float, float)  # time, freq
    remove_point_requested = Signal(float)      # time
    modify_point_requested = Signal(float, float) # time, new_freq
    point_drag_started = Signal()
    point_drag_finished = Signal()
    selection_changed = Signal(float, float)
    
    set_region_voiced_requested = Signal(float, float) # start_time, end_time
    set_region_unvoiced_requested = Signal(float, float) # start_time, end_time

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(6)

        self.legend_widget = QtWidget()
        self.legend_widget.setStyleSheet(
            "background: #f4f1e8; border: 1px solid #c8c2b8; border-radius: 6px;"
        )
        self.legend_layout = QHBoxLayout(self.legend_widget)
        self.legend_layout.setContentsMargins(10, 8, 10, 8)
        self.legend_layout.setSpacing(12)
        self.layout.addWidget(self.legend_widget)

        # Main pyqtgraph PlotWidget
        self.plot_widget = pg.PlotWidget(
            viewBox=SelectionViewBox(
                self._update_drag_selection,
                self._handle_point_drag,
                self._get_zoom_anchor_y,
            )
        )
        self.layout.addWidget(self.plot_widget)
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=False, y=False)
        self.vb = self.plot_widget.getViewBox()
        self.vb.setDefaultPadding(0.0)
        self.plot_widget.setLimits(xMin=0.0)
        self.plot_widget.setXRange(0.0, 1.0, padding=0)
        self.plot_widget.setYRange(0.0, 1000.0, padding=0)
        self.plot_widget.setCursor(Qt.IBeamCursor)

        # Spectrogram Image
        self.img_item = pg.ImageItem()
        self.img_item.setOpacity(0.82)
        self.plot_widget.addItem(self.img_item)
        
        # Pitch curve
        self.pitch_item = pg.PlotDataItem(
            pen=pg.mkPen(color='#1f5cff', width=2),
            symbol='o',
            symbolSize=6,
            symbolBrush='#1f5cff',
            connect='finite' # Skip NaNs
        )
        self.plot_widget.addItem(self.pitch_item)
        self.selected_point_item = pg.ScatterPlotItem(
            size=11,
            pen=pg.mkPen("#111111", width=1.5),
            brush=pg.mkBrush("#ffcc00"),
        )
        self.plot_widget.addItem(self.selected_point_item)
        self.quantile_lines = {
            "p20": pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("#2a6fdb", width=1.5, style=Qt.DashLine)),
            "p50": pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("#c0392b", width=1.5, style=Qt.DashLine)),
            "p80": pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("#0f766e", width=1.5, style=Qt.DashLine)),
        }
        for line in self.quantile_lines.values():
            line.hide()
            self.plot_widget.addItem(line)
        
        # Region selection
        self.region_item = pg.LinearRegionItem(values=[0, 1], movable=True)
        self.region_item.setZValue(10)
        self.region_item.hide() # Hidden by default
        self.region_item.sigRegionChanged.connect(self._emit_region_changed)
        self.plot_widget.addItem(self.region_item)
        
        # Praat-like grayscale: dark energy on a light background.
        colormap = pg.ColorMap(
            pos=np.array([0.0, 1.0]),
            color=np.array([[255, 255, 255, 255], [0, 0, 0, 255]], dtype=np.ubyte),
        )
        self.img_item.setColorMap(colormap)
        
        # Custom ViewBox interactions
        self.plot_widget.scene().sigMouseClicked.connect(self.on_mouse_click)
        self.plot_widget.scene().sigMouseMoved.connect(self.on_mouse_move)
        
        self.dragging_point_time = None
        self.dragging_point_index = None
        self.selected_point_index = None
        self._cached_timestamps = np.array([])
        self._cached_pitch_values = np.array([])
        self.segment_items = []
        self.audio_end_time = 1.0
        self.segment_color_map = {
            0: QColor(255, 210, 80, 200),   # silence
            1: QColor(90, 90, 90, 200),     # voiceless
            2: QColor(60, 200, 120, 200),   # voiced
        }
        self._build_segment_legend()
        
        # Axis labels
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.setLabel('left', 'Frequency', units='Hz')
        self.plot_widget.getAxis('bottom').setTextPen('k')
        self.plot_widget.getAxis('bottom').setPen('k')
        self.plot_widget.getAxis('left').setTextPen('k')
        self.plot_widget.getAxis('left').setPen('k')
        self.region_item.setBrush(QBrush(QColor(30, 100, 255, 45)))
        self.region_item.setHoverBrush(QBrush(QColor(30, 100, 255, 70)))
        for line in self.region_item.lines:
            line.setPen(pg.mkPen('#1f5cff', width=2))

    def _build_segment_legend(self):
        title = QLabel("Segment colors:")
        title.setStyleSheet("color: #111111; font-weight: 700; background: transparent;")
        self.legend_layout.addWidget(title)
        items = [
            ("Silence", QColor(255, 233, 153)),
            ("Voiceless", QColor(128, 128, 128)),
            ("Voiced", QColor(70, 190, 110)),
        ]
        for text, color in items:
            swatch = QFrame()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(
                f"background-color: rgba({color.red()}, {color.green()}, {color.blue()}, 180);"
                "border: 1px solid #666666;"
            )
            label = QLabel(text)
            label.setStyleSheet("color: #111111; font-weight: 600; background: transparent;")
            self.legend_layout.addWidget(swatch)
            self.legend_layout.addWidget(label)

        divider = QLabel("|")
        divider.setStyleSheet("color: #666666; font-weight: 700; background: transparent; padding: 0 4px;")
        self.legend_layout.addWidget(divider)

        quantile_title = QLabel("F0 lines:")
        quantile_title.setStyleSheet("color: #111111; font-weight: 700; background: transparent;")
        self.legend_layout.addWidget(quantile_title)

        quantile_items = [
            ("20%", "#2a6fdb"),
            ("50%", "#c0392b"),
            ("80%", "#0f766e"),
        ]
        for text, color in quantile_items:
            line_swatch = QFrame()
            line_swatch.setFixedSize(22, 3)
            line_swatch.setStyleSheet(
                f"background-color: {color}; border: none; border-radius: 1px;"
            )
            line_label = QLabel(text)
            line_label.setStyleSheet("color: #111111; font-weight: 600; background: transparent;")
            self.legend_layout.addWidget(line_swatch)
            self.legend_layout.addWidget(line_label)
        self.legend_layout.addStretch()

    def show_region(self, show=True):
        if show:
            start, end = self.region_item.getRegion()
            if end <= start:
                default_end = min(self.audio_end_time, max(0.25, self.audio_end_time * 0.2))
                self.region_item.setRegion([0.0, default_end])
            self.region_item.show()
            self._emit_region_changed()
        else:
            self.region_item.hide()
            
    def get_selected_region(self):
        return self.region_item.getRegion()

    def _emit_region_changed(self):
        start, end = self.region_item.getRegion()
        self.selection_changed.emit(float(start), float(end))

    def _clamp_time(self, time_value):
        return float(np.clip(time_value, 0.0, self.audio_end_time))

    def _update_drag_selection(self, start_x, end_x, is_finish):
        start = self._clamp_time(min(start_x, end_x))
        end = self._clamp_time(max(start_x, end_x))
        if abs(end - start) < 1e-4:
            return
        self.region_item.setRegion([start, end])
        self.region_item.show()
        self.selection_changed.emit(start, end)

    def _find_point_near_scene_pos(self, scene_pos, max_distance_px=10.0):
        valid_mask = ~np.isnan(self._cached_pitch_values)
        if len(self._cached_timestamps) == 0 or not np.any(valid_mask):
            return None
        candidate_indices = np.flatnonzero(valid_mask)
        nearest_idx = None
        nearest_dist = None
        for idx in candidate_indices:
            point_scene = self.vb.mapViewToScene(pg.Point(float(self._cached_timestamps[idx]), float(self._cached_pitch_values[idx])))
            dist = (point_scene.x() - scene_pos.x()) ** 2 + (point_scene.y() - scene_pos.y()) ** 2
            if nearest_dist is None or dist < nearest_dist:
                nearest_dist = dist
                nearest_idx = int(idx)
        if nearest_idx is None or nearest_dist is None:
            return None
        if nearest_dist <= max_distance_px ** 2:
            return nearest_idx
        return None

    def _update_selected_point_visual(self):
        if self.selected_point_index is None or len(self._cached_timestamps) == 0:
            self.selected_point_item.setData([], [])
            return
        idx = int(self.selected_point_index)
        if idx < 0 or idx >= len(self._cached_timestamps) or np.isnan(self._cached_pitch_values[idx]):
            self.selected_point_item.setData([], [])
            return
        self.selected_point_item.setData(
            [float(self._cached_timestamps[idx])],
            [float(self._cached_pitch_values[idx])],
        )

    def _handle_point_drag(self, ev):
        start_idx = self._find_point_near_scene_pos(ev.buttonDownScenePos())
        if start_idx is None and self.dragging_point_index is None:
            return False
        if self.dragging_point_index is None:
            self.dragging_point_index = start_idx
            self.selected_point_index = start_idx
            self.point_drag_started.emit()

        current_view = self.vb.mapSceneToView(ev.scenePos())
        idx = int(self.dragging_point_index)
        time_val = float(self._cached_timestamps[idx])
        freq_val = max(0.0, float(current_view.y()))
        self.modify_point_requested.emit(time_val, freq_val)
        if ev.isFinish():
            self.dragging_point_index = None
            self.point_drag_finished.emit()
        return True

    def _get_zoom_anchor_y(self, anchor_x):
        valid_mask = ~np.isnan(self._cached_pitch_values)
        if len(self._cached_timestamps) == 0 or not np.any(valid_mask):
            return self.vb.viewRange()[1][0] + (self.vb.viewRange()[1][1] - self.vb.viewRange()[1][0]) / 2.0
        x_min, x_max = self.vb.viewRange()[0]
        visible_mask = valid_mask & (self._cached_timestamps >= x_min) & (self._cached_timestamps <= x_max)
        if np.any(visible_mask):
            return float(np.nanmedian(self._cached_pitch_values[visible_mask]))
        nearest_idx = int(np.abs(self._cached_timestamps - anchor_x).argmin())
        nearest_val = self._cached_pitch_values[nearest_idx]
        if not np.isnan(nearest_val):
            return float(nearest_val)
        return float(np.nanmedian(self._cached_pitch_values[valid_mask]))

    def update_spectrogram(self, S_db, spec_times, spec_freqs):
        """
        Render spectrogram
        """
        if S_db is None:
            return
        
        max_db = float(np.nanmax(S_db))
        min_db = max_db - 50.0
        self.img_item.setImage(S_db.T, autoLevels=False)
        self.img_item.setLevels([min_db, max_db])
        
        # Set boundaries coordinates
        x_min, x_max = spec_times[0], spec_times[-1]
        self.audio_end_time = float(x_max)
        y_min, y_max = spec_freqs[0], spec_freqs[-1]
        self.img_item.setRect(QRectF(x_min, y_min, x_max - x_min, y_max - y_min))
        self.plot_widget.setLimits(xMin=0.0, xMax=x_max, yMin=y_min, yMax=y_max)
        self.plot_widget.setXRange(0.0, x_max, padding=0)
        self.plot_widget.setYRange(y_min, y_max, padding=0)
        default_end = min(x_max, max(0.25, x_max * 0.2))
        self.region_item.setBounds([0.0, x_max])
        self.region_item.setRegion([0.0, default_end])
        self.region_item.show()
        self.selection_changed.emit(0.0, default_end)

    def update_segments(self, timestamps, segment_labels):
        for item in self.segment_items:
            self.plot_widget.removeItem(item)
        self.segment_items.clear()

        if len(timestamps) == 0 or len(segment_labels) == 0:
            return

        view_range = self.vb.viewRange()
        y_min, y_max = view_range[1]
        if y_max <= y_min:
            y_min, y_max = 0.0, 1.0
        band_height = max((y_max - y_min) * 0.035, 40.0)
        band_bottom = y_max - band_height

        if len(timestamps) > 1:
            step = float(np.median(np.diff(timestamps)))
        else:
            step = 0.01
        half_width = step / 2.0

        start_idx = 0
        while start_idx < len(segment_labels):
            label = int(segment_labels[start_idx])
            end_idx = start_idx + 1
            while end_idx < len(segment_labels) and int(segment_labels[end_idx]) == label:
                end_idx += 1

            x0 = float(timestamps[start_idx] - half_width)
            x1 = float(timestamps[end_idx - 1] + half_width)
            if start_idx == 0:
                x0 = 0.0
            if end_idx == len(segment_labels):
                x1 = self.audio_end_time
            x0 = max(0.0, x0)
            x1 = min(self.audio_end_time, x1)
            rect = QGraphicsRectItem(QRectF(x0, band_bottom, x1 - x0, band_height))
            rect.setPen(QPen(Qt.NoPen))
            rect.setBrush(QBrush(self.segment_color_map.get(label, QColor(0, 0, 0, 0))))
            rect.setZValue(8)
            self.plot_widget.addItem(rect)
            self.segment_items.append(rect)
            start_idx = end_idx

    def update_pitch(self, timestamps, pitch_values):
        """
        Update the pitch contour
        """
        self._cached_timestamps = np.asarray(timestamps, dtype=float)
        self._cached_pitch_values = np.asarray(pitch_values, dtype=float)
        self.pitch_item.setData(x=timestamps, y=pitch_values)
        self._update_selected_point_visual()

    def update_quantile_lines(self, p20, p50, p80):
        for key, value in (("p20", p20), ("p50", p50), ("p80", p80)):
            line = self.quantile_lines[key]
            if np.isnan(value) or value <= 0:
                line.hide()
            else:
                line.setValue(float(value))
                line.show()

    def fit_to_audio(self):
        self.plot_widget.setXRange(0.0, self.audio_end_time, padding=0)

    def on_mouse_click(self, event):
        if not self.vb.sceneBoundingRect().contains(event.scenePos()):
            return
            
        mouse_point = self.vb.mapSceneToView(event.scenePos())
        t = mouse_point.x()
        f = mouse_point.y()

        if event.modifiers() == Qt.NoModifier and event.button() == Qt.LeftButton:
            selected_idx = self._find_point_near_scene_pos(event.scenePos())
            self.selected_point_index = selected_idx
            self._update_selected_point_visual()
            if selected_idx is not None:
                event.accept()
                return
            
        # Alt + Click: Add point (request snapping via signal)
        if event.modifiers() == Qt.AltModifier and event.button() == Qt.LeftButton:
            self.add_point_requested.emit(t, f)
            event.accept()
            return
            
        # Shift + Click: Remove point
        if event.modifiers() == Qt.ShiftModifier and event.button() == Qt.LeftButton:
            self.remove_point_requested.emit(t)
            event.accept()
            return

    def on_mouse_move(self, event):
        # We can implement live dragging of points here by detecting mouse drag.
        pass
        # A simple implementation can just rely on adding/removing for now. 
        # Dragging requires overriding mouseDragEvent which gets complex with pyqtgraph's default viewbox.
