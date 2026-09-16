"""
experiments/example_timelapse.py

First example experiment: record a simple single-channel timelapse
while displaying a flat/neutral phase mask on the SLM, and save the
result as a TIFF.

This script should stay minimal — only parameters and function
calls. All actual logic lives in core/ and functions/. Run it with
Micro-Manager (and the SLM's virtual/secondary display) already
running:

    python -m experiments.example_timelapse
"""
from pycromanager import Core

from core.config import load_config
from core.session import Session
from core.logging_setup import setup_logger
from core.file_io import save_timelapse_tiff
from functions.slm import SLMDisplay, flat_mask
from functions.acquisition import run_timelapse

# ---- experiment parameters ----
NUM_FRAMES = 50
INTERVAL_S = 1
EXPOSURE_MS = 50
LASER_POWER_MW = 5.0


def main():
    config = load_config()
    session = Session(config, script_name="example_timelapse")
    logger = setup_logger(session.path)
    logger.info(f"Run folder: {session.path}")

    core = Core()
    logger.info("Connected to Micro-Manager.")

    with SLMDisplay(config) as slm:
        slm.show_mask(flat_mask(config))
        logger.info("Flat phase mask displayed on SLM.")

        frames, metadata = run_timelapse(
            core, config, logger,
            num_frames=NUM_FRAMES,
            interval_s=INTERVAL_S,
            exposure_ms=EXPOSURE_MS,
            laser_power_mW=LASER_POWER_MW,
        )

    out_path = save_timelapse_tiff(session.path, frames, metadata)
    logger.info(f"Timelapse saved to {out_path}")


if __name__ == "__main__":
    main()
