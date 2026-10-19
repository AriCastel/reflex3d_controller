"""
experiments/timelapse.py

Continuous timelapse at the current position: a given number of frames
with no wait between them, at a given exposure (ms) and laser power (mW),
saved as <name>.tif (ImageJ TYX stack) in a run folder
<root_folder>/YYYYMMDD/YYYYMMDD_<name>_N/, together with <name>_metadata.json (settings, frames acquired, measured sampling rate,
per-frame camera timestamps, camera ROI, phase mask) and the phase mask
itself (slm_mask.tif + slm_mask_metadata.json).

Only a time loop goes to the MDA engine - no XYZ positions, channel
groups or pixel-size calibration - so the stage, which on this setup
must only be moved through its 'Current X/Y/Z' properties, is never
touched.

Frames have the camera's current ROI, so crop the sensor first (e.g.
napari-micromanager's Camera ROI tool) for higher frame rates. They're
kept in RAM during the run and the TIFF is written once it's over (as
Micro-Manager does for fast acquisitions), so disk writes can't slow the
camera down; a run that wouldn't fit in free RAM is refused before it
starts. BUFFER = "disk" (or acquisition_defaults.timelapse.buffer)
streams frames to a file on disk instead, for runs too long for RAM.

Two ways to run it:

* From the napari GUI (experiments/imaging_gui.py): fill in the
  'Timelapse' panel and press Start. The mask is whichever one was last
  loaded from the 'SLM phase mask' panel; the GUI's own viewer shows the
  frames as they come in.
* Headless (replaces example_timelapse_mda.py): set the parameters below
  and run
      python -m experiments.timelapse <name>
  This loads the hardware itself and shows PHASE_MASK for the run, with
  an optional napari window of the latest frame (PREVIEW). Closing that
  window before the end stops the run; the frames so far are kept.
"""
import re
import sys
import time

import numpy as np

from core.config import load_config
from core.file_io import TimelapseRamBuffer, TimelapseTiffStream, save_phase_mask
from core.logging_setup import setup_logger
from core.mmcore import load_mmcore
from core.session import Session
from functions.acquisition import sampling_rate_hz, start_continuous_timelapse
from functions.imaging_masks import (
    build_imaging_mask,
    describe_imaging_mask,
    remember_shown_mask,
    shown_mask,
)
from functions.mmcore_slm import MMCoreSLM

# ---- experiment parameters (None -> config defaults) ----
NAME = None            # None -> first command-line argument
NUM_FRAMES = None      # None -> acquisition_defaults.timelapse.num_frames
EXPOSURE_MS = None     # None -> acquisition_defaults.timelapse.exposure_ms
LASER_POWER_MW = None  # None -> laser.default_power_mW
# Headless only:
PHASE_MASK = "flat"    # "flat", "aberration_correction", "edof", "aberration_correction_edof",
                       # or None to leave the SLM as it is
PREVIEW = True         # napari window with the latest frame
# Both:
BUFFER = None          # "ram" or "disk"; None -> acquisition_defaults.timelapse.buffer ("ram")


def safe_name(name):
    """A timelapse name usable in folder and file names."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name or "").strip()).strip("_")
    if not cleaned:
        raise ValueError(f"Give the timelapse a name (got {name!r}).")
    return cleaned


def main(name=None, num_frames=None, exposure_ms=None, laser_power_mW=None,
         mmc=None, on_finished=None):
    """
    Headless when `mmc` is None: loads the hardware, shows PHASE_MASK and
    returns when the run is over. With `mmc` (the napari GUI's core): uses
    the mask already on the SLM, returns as soon as the run has started,
    and calls `on_finished(summary)` at the end. Returns the run folder.
    """
    name = safe_name(name or (sys.argv[1] if len(sys.argv) > 1 and mmc is None else None)
                     or NAME)
    config = load_config()
    defaults = config.get("acquisition_defaults", "timelapse", default={})
    num_frames = int(num_frames or NUM_FRAMES or defaults.get("num_frames", 100))
    exposure_ms = float(exposure_ms or EXPOSURE_MS or defaults.get("exposure_ms", 50))
    if laser_power_mW is None:
        laser_power_mW = LASER_POWER_MW
    if laser_power_mW is None:
        laser_power_mW = config.get("laser", "default_power_mW", default=0.0)
    headless = mmc is None

    buffer = (BUFFER or defaults.get("buffer", "ram")).lower()
    if buffer not in ("ram", "disk"):
        raise ValueError(f"buffer must be 'ram' or 'disk', got {buffer!r}")

    mask = info = None
    if headless and PHASE_MASK:
        mask, info = build_imaging_mask(config, PHASE_MASK)  # fails before touching hardware
    if headless:
        mmc = load_mmcore(config)

    # Frames have the camera's CURRENT ROI - whatever was cropped in the GUI
    # (napari-micromanager's Camera ROI tool sets it on this same core); it's
    # never changed here.
    roi = list(mmc.getROI())
    frame_shape = (mmc.getImageHeight(), mmc.getImageWidth())
    dtype = {1: np.uint8, 2: np.uint16, 4: np.uint32}.get(mmc.getBytesPerPixel(), np.uint16)
    if buffer == "ram":
        TimelapseRamBuffer.check_fits(num_frames, frame_shape, dtype)  # before any folder

    session = Session(config, script_name=name)
    logger = setup_logger(session.path)
    logger.info(f"Run folder: {session.path}")

    if headless:
        logger.info("Connected to Micro-Manager via pymmcore-plus.")
        if mask is not None:
            with MMCoreSLM(mmc, config) as slm:
                slm.show_mask(mask)
            remember_shown_mask(mask, info)
            time.sleep(config.get("slm", "settle_frames", default=3)
                       / config.get("slm", "refresh_rate_hz", default=30))

    mask, info = shown_mask()
    if info is None:
        info = {"preset": None, "note": "the SLM mask wasn't set by this program - unknown"}
    else:
        save_phase_mask(session.path, mask, info)
    logger.info(describe_imaging_mask(info) if info.get("channels") else info["note"])

    writer = TimelapseRamBuffer if buffer == "ram" else TimelapseTiffStream
    stream = writer(session.path, f"{name}.tif", num_frames, frame_shape, dtype,
                    frame_interval_s=exposure_ms / 1000.0)
    logger.info(f"Camera ROI (x, y, w, h) = {roi}; frames buffered in "
                + ("RAM, written to disk when the run ends" if buffer == "ram"
                   else "a file on disk as they arrive"))
    metadata = {
        "name": name,
        "num_frames": num_frames,
        "exposure_ms": exposure_ms,
        "laser_power_mW": laser_power_mW,
        "interval_s": 0.0,
        "max_sampling_rate_hz": sampling_rate_hz(exposure_ms),
        "camera_roi_xywh": roi,
        "buffer": buffer,
        "core_sequence_buffer_mb": mmc.getCircularBufferMemoryFootprint(),
        "phase_mask": info,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    def finished(summary):
        path = stream.close({**metadata, **summary,
                             "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        rate = summary["measured_sampling_rate_hz"]
        logger.info(f"{summary['frames_acquired']}/{num_frames} frames saved to {path}"
                    + (f"; measured {rate:.2f} Hz" if rate else ""))
        if on_finished is not None:
            on_finished({**summary, "path": path})

    block = headless and not PREVIEW
    recorder = start_continuous_timelapse(mmc, config, logger, stream, num_frames, exposure_ms,
                                          laser_power_mW, block=block, on_finished=finished)

    if headless and PREVIEW:
        from functions.napari_preview import show_latest_frames

        show_latest_frames(lambda: recorder.latest, num_frames,
                           title=f"ReflEx3D — timelapse '{name}' (close to stop early)")
        if not recorder.done.is_set():
            logger.info("Preview closed before the end - stopping the timelapse.")
            mmc.mda.cancel()
        recorder.done.wait()
    return session.path


if __name__ == "__main__":
    main()
