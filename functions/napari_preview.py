"""
functions/napari_preview.py

napari viewers for the pymmcore-plus stack:

* `show_latest_frames` shows the most recent frame of a running
  acquisition (headless timelapse preview).
* `run_live_roi_preview` streams the full camera frame live, with the
  per-channel ROIs outlined, so the sample can be positioned before
  an acquisition.
* `show_verification_frame` shows one snapped frame with the localized
  point source marked.
* `add_script_panel` docks a panel of buttons that run scripts inside
  the GUI's process (e.g. experiments/load_phase_mask.py).
* `add_timelapse_panel` docks the simplified timelapse form (name,
  exposure, laser power, frames, with the resulting sampling rate).

`run_live_roi_preview`, `show_verification_frame` and
`show_latest_frames` block until the scientist closes the viewer window.
"""
import numpy as np

ROI_COLORS = ["magenta", "cyan", "yellow", "lime"]


def run_live_roi_preview(mmc, rois, title="Live preview", poll_interval_ms=50):
    """
    Stream the full camera frame live into a napari viewer, with each
    channel ROI outlined and labelled, and block until the viewer
    window is closed.

    `rois` maps a channel name to (x, y, width, height) in full-frame
    camera pixels. The camera ROI is cleared first so the whole sensor
    is shown. Illumination (laser, SLM mask) is the caller's job.
    """
    import napari
    from qtpy.QtCore import QTimer

    mmc.clearROI()
    height, width = mmc.getImageHeight(), mmc.getImageWidth()

    viewer = napari.Viewer(title=title)
    layer = viewer.add_image(
        np.zeros((height, width), dtype=np.uint16), name="Live (full frame)"
    )

    names = list(rois)
    colors = [ROI_COLORS[i % len(ROI_COLORS)] for i in range(len(names))]
    # napari shapes are (row, col) = (y, x)
    boxes = [[[y, x], [y + h, x + w]] for (x, y, w, h) in rois.values()]
    viewer.add_shapes(
        boxes,
        shape_type="rectangle",
        edge_color=colors,
        face_color="transparent",
        edge_width=max(2, round(min(height, width) / 250)),
        features={"channel": names, "color": colors},
        text={
            "string": "{channel}",
            "anchor": "upper_left",
            "size": 14,
            "color": {"feature": "color"},
        },
        name="Channel ROIs",
    )

    contrast_set = False

    def _update():
        nonlocal contrast_set
        if mmc.getRemainingImageCount() == 0:
            return
        layer.data = mmc.getLastImage()
        if not contrast_set:
            layer.reset_contrast_limits()
            contrast_set = True

    timer = QTimer()
    timer.timeout.connect(_update)
    mmc.startContinuousSequenceAcquisition(0)
    timer.start(poll_interval_ms)
    try:
        napari.run()
    finally:
        timer.stop()
        mmc.stopSequenceAcquisition()


def show_verification_frame(frame, loc, title="Point-source verification"):
    """
    Show one frame with the localized point source marked (or a note
    that nothing was localized), and block until the viewer is closed.
    """
    import napari

    viewer = napari.Viewer(title=title)
    viewer.add_image(frame, name="Verification frame")
    viewer.text_overlay.visible = True
    if loc is None:
        viewer.text_overlay.text = "NO point source localized"
    else:
        # An outlined circle, so the marker surrounds the bead instead
        # of hiding it.
        viewer.add_points(
            [[loc["y"], loc["x"]]],
            symbol="disc",
            face_color="transparent",
            border_color="red",
            border_width=0.06,
            size=max(12, min(frame.shape) / 6),
            name="Localized source",
        )
        viewer.text_overlay.text = (
            f"Source at ({loc['x']:.1f}, {loc['y']:.1f}) px, "
            f"SNR {loc['snr']:.1f}"
        )
    napari.run()


def add_script_panel(viewer, title, actions):
    """
    Dock a panel with one button per entry of `actions` (label ->
    callable, run on the Qt main thread when clicked). A failing action
    shows a napari error notification instead of taking the GUI down,
    and the panel's status line shows the last action and its outcome.
    Returns the panel widget.
    """
    from napari.utils.notifications import show_error, show_info
    from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

    panel = QWidget()
    layout = QVBoxLayout(panel)
    status = QLabel("")
    status.setWordWrap(True)

    for label, action in actions.items():
        button = QPushButton(label)

        def _run(_checked=False, label=label, action=action):
            try:
                action()
            except Exception as exc:
                status.setText(f"FAILED: {label}")
                show_error(f"{label} failed: {exc}")
            else:
                status.setText(f"Last: {label}")
                show_info(f"{label}: done")

        button.clicked.connect(_run)
        layout.addWidget(button)

    layout.addWidget(status)
    panel.setMaximumHeight(panel.sizeHint().height() + 40)  # leave the rest to other docks
    viewer.window.add_dock_widget(panel, name=title, area="right")
    return panel


def show_latest_frames(latest, total, title="Timelapse", poll_interval_ms=100):
    """
    A napari window showing the most recent frame of a running
    acquisition: `latest()` returns (index, frame) or None. Only one frame
    is held, whatever the length of the run. Blocks until the window is
    closed.
    """
    import napari
    from qtpy.QtCore import QTimer

    viewer = napari.Viewer(title=title)
    viewer.text_overlay.visible = True
    viewer.text_overlay.text = "waiting for the first frame..."
    state = {"layer": None, "shown": None}

    def _update():
        item = latest()
        if item is None or item[0] == state["shown"]:
            return
        t, frame = item
        if state["layer"] is None:
            state["layer"] = viewer.add_image(frame, name="Latest frame")
        else:
            state["layer"].data = frame
        state["shown"] = t
        viewer.text_overlay.text = f"frame {t + 1}/{total}"

    timer = QTimer()
    timer.timeout.connect(_update)
    timer.start(poll_interval_ms)
    try:
        napari.run()
    finally:
        timer.stop()


def _format_duration(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    if h:
        return f"{h} h {m} min"
    if m:
        return f"{m} min {s:.0f} s"
    return f"{s:.2f} s" if s < 10 else f"{s:.0f} s"


def add_timelapse_panel(viewer, start, stop, exposure_ms, laser_power_mW, num_frames,
                        max_laser_mW=None, title="Timelapse"):
    """
    Dock the simplified timelapse form: name, exposure (ms), laser power
    (mW) and number of frames, with the resulting sampling rate (1 /
    exposure, an upper bound) and minimum duration updated as they're
    edited.

    Start calls `start(name, exposure_ms, laser_power_mW, num_frames,
    done)`; `start` must return quickly and call `done(message)` - from
    any thread - when the run ends, which re-enables the form. Stop calls
    `stop()`. Returns the panel widget.
    """
    from napari.utils.notifications import show_error
    from qtpy.QtCore import QObject, Signal, Slot
    from qtpy.QtWidgets import (
        QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
        QSpinBox, QVBoxLayout, QWidget,
    )

    class _Relay(QObject):
        # Lives on the GUI thread (created here, parented to the panel), and
        # its own slot is the receiver: emitting `finished` from any other
        # thread is then queued onto the GUI thread, where widgets may be
        # touched. (A plain-function receiver would run in the emitting thread.)
        finished = Signal(str)

        def __init__(self, parent, callback):
            super().__init__(parent)
            self._callback = callback
            self.finished.connect(self._deliver)

        @Slot(str)
        def _deliver(self, message):
            self._callback(message)

    from functions.acquisition import sampling_rate_hz

    panel = QWidget()
    outer = QVBoxLayout(panel)
    form = QFormLayout()
    outer.addLayout(form)

    name = QLineEdit()
    name.setPlaceholderText("e.g. beads_edof")
    exposure = QDoubleSpinBox()
    exposure.setRange(0.01, 100000.0); exposure.setDecimals(2); exposure.setSuffix(" ms")
    exposure.setValue(exposure_ms)
    laser = QDoubleSpinBox()
    laser.setRange(0.0, float(max_laser_mW or 10000.0)); laser.setDecimals(1); laser.setSuffix(" mW")
    laser.setValue(laser_power_mW)
    frames = QSpinBox()
    frames.setRange(1, 10_000_000); frames.setValue(num_frames)
    rate, duration = QLabel(), QLabel()
    rate.setToolTip("1 / exposure: the rate if the camera is exposure-limited. At short "
                    "exposures the readout (ROI size) can limit it; the measured rate is "
                    "reported after the run.")
    for label, widget in (("Name", name), ("Exposure", exposure), ("Laser power", laser),
                          ("Frames", frames), ("Sampling rate", rate), ("Duration", duration)):
        form.addRow(label, widget)

    buttons = QHBoxLayout()
    start_button, stop_button = QPushButton("Start timelapse"), QPushButton("Stop")
    stop_button.setEnabled(False)
    buttons.addWidget(start_button); buttons.addWidget(stop_button)
    outer.addLayout(buttons)
    status = QLabel("")
    status.setWordWrap(True)
    outer.addWidget(status)
    outer.addStretch()
    inputs = (name, exposure, laser, frames)

    def _update_rate(*_):
        hz = sampling_rate_hz(exposure.value())
        rate.setText(f"<b>{hz:.2f} Hz</b> (max)")
        duration.setText(f"≥ {_format_duration(frames.value() / hz)}")

    def _set_running(running):
        start_button.setEnabled(not running)
        stop_button.setEnabled(running)
        for w in inputs:
            w.setEnabled(not running)

    def _on_finished(message):
        _set_running(False)
        status.setText(message)

    relay = _Relay(panel, _on_finished)
    _done = relay.finished.emit

    def _start(_checked=False):
        if not name.text().strip():
            status.setText("Give the timelapse a name first.")
            return
        _set_running(True)
        status.setText("Running...")
        try:
            start(name.text().strip(), exposure.value(), laser.value(), frames.value(), _done)
        except Exception as exc:
            _set_running(False)
            status.setText(f"FAILED: {exc}")
            show_error(f"Timelapse failed to start: {exc}")

    def _stop(_checked=False):
        status.setText("Stopping...")
        stop()

    exposure.valueChanged.connect(_update_rate)
    frames.valueChanged.connect(_update_rate)
    start_button.clicked.connect(_start)
    stop_button.clicked.connect(_stop)
    _update_rate()

    panel.setMinimumHeight(outer.minimumSize().height() + 60)  # don't let the dock squash the rows
    viewer.window.add_dock_widget(panel, name=title, area="right")
    return panel
