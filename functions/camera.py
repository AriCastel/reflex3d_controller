"""
functions/camera.py

General-purpose camera control via the Micro-Manager core. Never
writes to disk — that's core/file_io.py's job.
"""
import numpy as np


def set_exposure(core, exposure_ms):
    core.set_exposure(exposure_ms)


def snap_image(core):
    """Snap a single image and return it as a 2D numpy array (Y, X)."""
    core.snap_image()
    tagged_image = core.get_tagged_image()
    width = tagged_image.tags["Width"]
    height = tagged_image.tags["Height"]
    return np.reshape(tagged_image.pix, (height, width))
