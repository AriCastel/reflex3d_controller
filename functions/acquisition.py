"""
functions/acquisition.py

Orchestration routines that combine the general-purpose hardware
functions (laser, camera, ...) into full acquisition sequences.
Still never writes to disk directly — returns frames + metadata for
the experiment script to hand to core/file_io.py.

`run_timelapse` below is the legacy, pycromanager-based orchestration
routine. `build_timelapse_sequence`/`run_timelapse_mda` are the
pymmcore-plus equivalent, driving the acquisition through pymmcore-plus's
MDA engine instead of a manual snap loop.
"""
import time
from datetime import datetime

import numpy as np

from functions.camera import set_exposure, snap_image
from functions.laser import laser_off, laser_on, set_laser_power


def run_timelapse(core, config, logger, num_frames, interval_s, exposure_ms, laser_power_mW):
    """
    (legacy, pycromanager) Acquire a simple single-channel, single-plane
    timelapse: laser on, snap `num_frames` images spaced `interval_s`
    apart, laser off.

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


def build_timelapse_sequence(num_frames, interval_s):
    """
    Build a pymmcore-plus MDA (useq-schema) sequence for a simple
    single-position, single-channel, single-plane timelapse:
    `num_frames` timepoints spaced `interval_s` apart.

    Exposure and laser power aren't part of the sequence itself here
    (kept symmetric with the legacy `run_timelapse`, which sets them
    once up front) - see `run_timelapse_mda`.
    """
    from useq import MDASequence

    return MDASequence(time_plan={"interval": interval_s, "loops": num_frames})


def run_timelapse_mda(mmc, config, logger, num_frames, interval_s, exposure_ms,
                       laser_power_mW, viewer=None):
    """
    Acquire the same simple timelapse as `run_timelapse`, but driven by
    pymmcore-plus's MDA engine (`mmc.run_mda`) instead of a manual snap
    loop.

    Without a `viewer`, blocks until the sequence finishes (mirrors
    `run_timelapse`). With a `viewer` (a napari.Viewer), attaches a
    live-updating preview layer (functions/napari_preview.py) and runs
    the MDA on a background thread instead, since the caller still
    needs the main thread free to call `napari.run()` and pump the Qt
    event loop while frames come in. Either way, the laser is switched
    off when the sequence finishes, via the MDA engine's
    `sequenceFinished` event rather than a `try`/`finally`.

    Returns
    -------
    sequence : useq.MDASequence
        The sequence that was run/started, for the caller to fold into
        saved metadata.
    """
    from functions.mmcore_laser import laser_off_mmcore, laser_on_mmcore, set_laser_power_mmcore
    from functions.napari_preview import attach_live_preview

    mmc.setExposure(exposure_ms)
    set_laser_power_mmcore(mmc, config, laser_power_mW)
    laser_on_mmcore(mmc, config)

    sequence = build_timelapse_sequence(num_frames, interval_s)
    if viewer is not None:
        attach_live_preview(mmc, viewer, sequence)

    def _on_finished(*_):
        laser_off_mmcore(mmc, config)
        logger.info(f"MDA timelapse finished ({num_frames} frames, {interval_s}s interval).")

    mmc.mda.events.sequenceFinished.connect(_on_finished)
    mmc.run_mda(sequence, block=viewer is None)

    return sequence
