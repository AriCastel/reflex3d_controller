"""
functions/napari_preview.py

napari viewers for the pymmcore-plus stack:

* `attach_live_preview` keeps a napari image layer in sync with frames
  as CMMCorePlus's MDA engine (`mmc.mda`) acquires them.
* `run_live_roi_preview` streams the full camera frame live, with the
  per-channel ROIs outlined, so the sample can be positioned before
  an acquisition.
* `show_verification_frame` shows one snapped frame with the localized
  point source marked.

The last two block until the scientist closes the viewer window.
"""
import numpy as np
from superqt.utils import ensure_main_thread

ROI_COLORS = ["magenta", "cyan", "yellow", "lime"]


def attach_live_preview(mmc, viewer, sequence, layer_name="MDA preview"):
    """
    Add a napari image layer sized to `sequence` and wire it up to
    update live as `mmc` acquires each MDA frame.

    `mmc.mda`'s frameReady signal fires on the acquisition thread, so
    the layer update is marshalled onto the Qt main thread via
    `superqt.utils.ensure_main_thread` before touching napari.

    Returns the napari layer.
    """
    num_frames = sequence.sizes.get("t", 1)
    height = mmc.getImageHeight()
    width = mmc.getImageWidth()
    layer = viewer.add_image(
        np.zeros((num_frames, height, width), dtype=np.uint16),
        name=layer_name,
    )

    @ensure_main_thread
    def _update(image, event, meta):
        t_index = event.index.get("t", 0)
        layer.data[t_index] = image
        layer.refresh()

    mmc.mda.events.frameReady.connect(_update)
    return layer


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
