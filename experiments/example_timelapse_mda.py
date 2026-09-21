"""
experiments/example_timelapse_mda.py

pymmcore-plus / napari equivalent of example_timelapse.py: record the
same simple single-channel timelapse (flat/neutral SLM mask, laser on
for the duration) but drive the acquisition through pymmcore-plus's
MDA engine and watch it live in a napari viewer, instead of only
inspecting the saved TIFF afterwards.

This script should stay minimal — only parameters and function calls.
All actual logic lives in core/ and functions/. Run it with a real
Micro-Manager device adapter + system config pointed to by
`pymmcore_plus.device_adapter_path` / `pymmcore_plus.system_config_path`
in config/microscope_config.json (and the SLM's virtual/secondary
display already available):

    python -m experiments.example_timelapse_mda

The original pycromanager-driven script (experiments/example_timelapse.py)
is kept as-is for the existing MM-GUI-driven workflow.
"""
import napari

from core.config import load_config
from core.session import Session
from core.logging_setup import setup_logger
from core.mmcore import load_mmcore
from functions.slm import SLMDisplay, flat_mask
from functions.acquisition import run_timelapse_mda

# ---- experiment parameters ----
NUM_FRAMES = 50
INTERVAL_S = 0
EXPOSURE_MS = 100
LASER_POWER_MW = 50.0


def main():
    config = load_config()
    session = Session(config, script_name="example_timelapse_mda")
    logger = setup_logger(session.path)
    logger.info(f"Run folder: {session.path}")

    mmc = load_mmcore(config)
    logger.info("Connected to Micro-Manager via pymmcore-plus.")

    viewer = napari.Viewer(title="ReflEx3D — live timelapse preview")

    with SLMDisplay(config) as slm:
        slm.show_mask(flat_mask(config))
        logger.info("Flat phase mask displayed on SLM.")

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
