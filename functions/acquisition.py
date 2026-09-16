"""
functions/acquisition.py

Orchestration routines that combine the general-purpose hardware
functions (laser, camera, ...) into full acquisition sequences.
Still never writes to disk directly — returns frames + metadata for
the experiment script to hand to core/file_io.py.
"""
import time
from datetime import datetime

import numpy as np

from functions.camera import set_exposure, snap_image
from functions.laser import laser_off, laser_on, set_laser_power


def run_timelapse(core, config, logger, num_frames, interval_s, exposure_ms, laser_power_mW):
    """
    Acquire a simple single-channel, single-plane timelapse: laser on,
    snap `num_frames` images spaced `interval_s` apart, laser off.

    Returns
    -------
    frames : np.ndarray, shape (T, Y, X)
    metadata : dict
    """
    set_exposure(core, exposure_ms)
    set_laser_power(core, config, laser_power_mW)
    laser_on(core, config)

    frames = []
    timestamps = []

    try:
        for i in range(num_frames):
            t0 = time.time()

            frame = snap_image(core)
            frames.append(frame)
            timestamps.append(datetime.now().isoformat())
            logger.info(f"Frame {i + 1}/{num_frames} acquired.")

            if i < num_frames - 1:
                elapsed = time.time() - t0
                time.sleep(max(0.0, interval_s - elapsed))
    finally:
        laser_off(core, config)

    metadata = {
        "num_frames": num_frames,
        "interval_s": interval_s,
        "exposure_ms": exposure_ms,
        "laser_power_mW": laser_power_mW,
        "timestamps": timestamps,
        "slm_mask_note": "flat/neutral phase mask",
    }
    return np.stack(frames, axis=0), metadata
