"""
experiments/mm2_slm_aberration_correction_edof.py (Micro-Manager 2.0 GUI stack, via pycromanager)

Load the stored aberration correction PLUS the extended-depth-of-field
primary spherical (imaging_masks.edof_zernike_rad_rms, added to the
correction's own Noll 11 term) onto both channels' pupils on the SLM.

Micro-Manager 2.0 version of experiments/slm_aberration_correction_edof.py, for when the
microscope is run from a Micro-Manager 2.0 instance instead of the
napari GUI. With Micro-Manager open (hardware config including the
Generic SLM device, slm.device_label), run from a terminal:

    python -m experiments.mm2_slm_aberration_correction_edof

The mask is pushed through Micro-Manager's SLM device, so it stays on
the SLM after this script exits - until another mask is loaded. Load it
after positioning the sample and before starting the acquisition in
Micro-Manager. The mask and what it's made of are recorded in a small
run folder, so the data can be matched to the mask that was on the SLM.
"""
from pycromanager import Core

from core.config import load_config
from core.file_io import save_phase_mask
from core.logging_setup import setup_logger
from core.session import Session
from functions.imaging_masks import build_imaging_mask, describe_imaging_mask
from functions.mm2_slm import MM2SLM

# ---- parameters ----
PRESET = "aberration_correction_edof"
CHANNELS = None  # None -> imaging_masks.channels from the config (both channels)


def main():
    config = load_config()
    mask, info = build_imaging_mask(config, PRESET, CHANNELS)  # fails before touching the SLM

    slm = MM2SLM(Core(), config)
    slm.show_mask(mask)

    session = Session(config, script_name="mm2_slm_aberration_correction_edof")
    logger = setup_logger(session.path)
    save_phase_mask(session.path, mask, info)
    logger.info(describe_imaging_mask(info))
    logger.info(f"Shown on the SLM through Micro-Manager; recorded in {session.path}")
    return info


if __name__ == "__main__":
    main()
