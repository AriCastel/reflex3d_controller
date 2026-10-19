"""
experiments/load_phase_mask.py

Load an imaging phase mask onto the SLM: one of functions/imaging_masks.py's
presets, on both channels' pupils (imaging_masks.channels):

    flat                        completely flat (neutral)
    aberration_correction       the stored correction (config/phase_masks.json,
                                method imaging_masks.aberration_correction_method)
    edof                        primary spherical (Noll 11) only, amplitude in
                                imaging_masks.edof_zernike_rad_rms
    aberration_correction_edof  both, added term by term

Load it after positioning the sample and before starting the acquisition.
The mask stays on the SLM until another one is loaded. Nothing is
acquired: the mask and what it's made of are recorded in a small run
folder, so the data can be matched to the mask that was on the SLM.

The same script works with both ways of running the microscope:

* napari GUI (experiments/imaging_gui.py): use the 'SLM phase mask' panel,
  or napari's console:   %run experiments/load_phase_mask.py edof
  (it must run inside the GUI's process, which owns the hardware);
* Micro-Manager 2.0: with Micro-Manager open (pycro-manager server enabled,
  Tools > Options), from a terminal:   python -m experiments.load_phase_mask edof

The preset is taken from main()'s argument, else the command line, else
PRESET below. BACKEND picks where the SLM is found (None = the napari GUI
in this process if there is one, else Micro-Manager 2.0).
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
    PRESETS,
    build_imaging_mask,
    describe_imaging_mask,
    remember_shown_mask,
)
from functions.mmcore_slm import MMCoreSLM, find_slm_core

# ---- parameters ----
PRESET = None    # e.g. "edof"; None -> must be given as main()'s argument or on the command line
CHANNELS = None  # None -> imaging_masks.channels from the config (both channels)
BACKEND = None   # None (auto), "napari" or "micromanager"


def main(preset=None):
    preset = preset or (sys.argv[1] if len(sys.argv) > 1 else None) or PRESET
    if preset not in PRESETS:
        raise SystemExit(f"Choose a phase mask: python -m experiments.load_phase_mask "
                         f"<{'|'.join(PRESETS)}> (got {preset!r})")

    config = load_config()
    mask, info = build_imaging_mask(config, preset, CHANNELS)  # fails before touching the SLM
    core = find_slm_core(config, BACKEND)

    with MMCoreSLM(core, config) as slm:
        if slm.acquisition_running():
            raise RuntimeError("An acquisition is running - not changing the SLM mask mid-run.")
        slm.show_mask(mask)
    remember_shown_mask(mask, info)

    session = Session(config, script_name=f"load_phase_mask_{preset}")
    logger = setup_logger(session.path)
    save_phase_mask(session.path, mask, info)
    logger.info(describe_imaging_mask(info))
    logger.info(f"Shown on the SLM; recorded in {session.path}")
    return info


if __name__ == "__main__":
    main()
