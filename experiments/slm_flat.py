"""
experiments/slm_flat.py

Load a completely flat (neutral) phase mask onto the whole SLM.

Run it INSIDE the napari GUI's Python process, after positioning the
sample and before starting the acquisition: from the 'SLM phase mask'
panel of `python -m experiments.imaging_gui`, or from napari's console
with

    %run experiments/slm_flat.py

(Micro-Manager hardware can only be open in one process, and the SLM
only shows a mask while the process that loaded it is alive, so this
can't be a separate `python` command.) The mask stays up until another
mask is loaded or the GUI is closed. Nothing is acquired: the mask and
what it's made of are recorded in a small run folder, so the data can
be matched to the mask that was on the SLM.
"""
import sys
from pathlib import Path

if __package__ in (None, ""):  # run by path, e.g. %run in napari's console
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import load_config
from core.file_io import save_phase_mask
from core.logging_setup import setup_logger
from core.session import Session
from functions.imaging_masks import (
    build_imaging_mask,
    describe_imaging_mask,
    get_running_core,
    show_imaging_mask,
)
from functions.mmcore_slm import MMCoreSLM

# ---- parameters ----
PRESET = "flat"
CHANNELS = None  # None -> imaging_masks.channels from the config (both channels)


def main():
    config = load_config()
    mmc = get_running_core(config)
    mask, info = build_imaging_mask(config, PRESET, CHANNELS)

    with MMCoreSLM(mmc, config) as slm:
        show_imaging_mask(mmc, slm, mask)

    session = Session(config, script_name="slm_flat")
    logger = setup_logger(session.path)
    save_phase_mask(session.path, mask, info)
    logger.info(describe_imaging_mask(info))
    logger.info(f"Shown on the SLM; recorded in {session.path}")
    return info


if __name__ == "__main__":
    main()
