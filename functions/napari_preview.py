"""
functions/napari_preview.py

Live napari preview for pymmcore-plus MDA runs: keeps a napari image
layer in sync with frames as CMMCorePlus's MDA engine (`mmc.mda`)
acquires them. Used by the `*_mda` experiments in place of the legacy
save-then-inspect workflow, for real-time viewing during acquisition.
"""
import numpy as np
from superqt.utils import ensure_main_thread


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
