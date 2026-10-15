"""
experiments/example_timelapse_mda.py

pymmcore-plus / napari equivalent of example_timelapse.py: record the
same simple single-channel timelapse (laser on for the duration) but
drive the acquisition through pymmcore-plus's MDA engine and watch it
live in a napari viewer, instead of only inspecting the saved TIFF
afterwards.

PHASE_MASK picks what the SLM shows for the whole acquisition, one of
functions/imaging_masks.py's presets: "flat" (neutral, the default),
"aberration_correction" (the stored correction from
config/phase_masks.json), "edof" (primary spherical) or
"aberration_correction_edof" (both), on the channels in
PHASE_MASK_CHANNELS. The mask is built before any hardware is touched,
so a missing stored correction stops the run before it starts, and it
is saved into the run folder (slm_mask.tif + slm_mask_metadata.json).

This script should stay minimal — only parameters and function calls.
All actual logic lives in core/ and functions/. Run it with a real
Micro-Manager device adapter + system config pointed to by
`pymmcore_plus.device_adapter_path` / `pymmcore_plus.system_config_path`
in config/microscope_config.json, with the Generic SLM device
(`slm.device_label`) loaded in that system config:

    python -m experiments.example_timelapse_mda

The original pycromanager-driven script (experiments/example_timelapse.py)
is kept as-is for the existing MM-GUI-driven workflow, including its
plain-secondary-display SLM handling — the SLM here goes through
Micro-Manager instead (functions/mmcore_slm.py's MMCoreSLM).
"""
import time

import napari

from core.config import load_config
from core.session import Session
from core.logging_setup import setup_logger
from core.mmcore import load_mmcore
from core.file_io import save_phase_mask
from calibration.fourier_alignment import slm_settle_seconds
from functions.imaging_masks import build_imaging_mask, describe_imaging_mask
from functions.mmcore_slm import MMCoreSLM
from functions.acquisition import run_timelapse_mda

# ---- experiment parameters ----
NUM_FRAMES = 50
INTERVAL_S = 0
EXPOSURE_MS = 100
LASER_POWER_MW = 50.0
PHASE_MASK = "flat"  # "flat", "aberration_correction", "edof" or "aberration_correction_edof"
PHASE_MASK_CHANNELS = None  # None -> imaging_masks.channels from the config (both channels)


def main():
    config = load_config()
    mask, mask_info = build_imaging_mask(config, PHASE_MASK, PHASE_MASK_CHANNELS)

    session = Session(config, script_name="example_timelapse_mda")
    logger = setup_logger(session.path)
    logger.info(f"Run folder: {session.path}")
    save_phase_mask(session.path, mask, mask_info)
    logger.info(describe_imaging_mask(mask_info))

    mmc = load_mmcore(config)
    logger.info("Connected to Micro-Manager via pymmcore-plus.")

    viewer = napari.Viewer(title="ReflEx3D — live timelapse preview")

    with MMCoreSLM(mmc, config) as slm:
        slm.show_mask(mask)
        time.sleep(slm_settle_seconds(config))
        logger.info(f"Phase mask '{PHASE_MASK}' displayed on SLM.")

        sequence = run_timelapse_mda(
            mmc, config, logger,
            num_frames=NUM_FRAMES,
            interval_s=INTERVAL_S,
            exposure_ms=EXPOSURE_MS,
            laser_power_mW=LASER_POWER_MW,
            viewer=viewer,
        )
        logger.info(f"MDA sequence started: {sequence}")

        # Blocks until the viewer window is closed; the MDA itself runs
        # on a background thread and streams frames into the preview
        # layer as they're acquired (see functions/napari_preview.py).
        napari.run()


if __name__ == "__main__":
    main()
