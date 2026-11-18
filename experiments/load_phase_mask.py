"""
experiments/load_phase_mask.py

Load an imaging phase mask onto the SLM, on both channels' pupils
(imaging_masks.channels): one mask by name, alone or added to the stored
aberration correction (see functions/imaging_masks.py).

    flat             completely flat (neutral)
    edof             primary spherical (Noll 11), amplitude in
                     imaging_masks.edof_zernike_rad_rms
    annular_binary   concentric 0 / pi rings, outer radii (fractions of the
                     pupil radius) in imaging_masks.annular_binary_rho,
                     default [0.76, 0.93]
    <name>           any mask in the library, config/mask_library/<name>.json
                     (e.g. zoned_focus_3, zoned_focus_4_sweep, spherical_defocus;
                     run with no argument to list what is there)

With the aberration correction (config/phase_masks.json, method
imaging_masks.aberration_correction_method): add --aberration-correction
(-a), or prefix the name:

    aberration_correction          the stored correction alone
    aberration_correction_<mask>   the correction plus that mask

Load it after positioning the sample and before starting the acquisition.
The mask stays on the SLM until another one is loaded. Nothing is
acquired and nothing is written to disk (swapping masks is routine, so no
run folder per load): the timelapse and the other experiments record the
mask they run with in their own folders.

The same script works with both ways of running the microscope:

* napari GUI (experiments/imaging_gui.py): use the 'SLM phase mask' panel,
  or napari's console:   %run experiments/load_phase_mask.py zoned_focus_3 -a
  (it must run inside the GUI's process, which owns the hardware);
* Micro-Manager 2.0: with Micro-Manager open (pycro-manager server enabled,
  Tools > Options), from a terminal:   python -m experiments.load_phase_mask zoned_focus_3 -a

The mask is taken from main()'s arguments, else the command line, else
PRESET and ABERRATION_CORRECTION below. BACKEND picks where the SLM is found (None = the napari GUI
in this process if there is one, else Micro-Manager 2.0).
"""
import sys
from pathlib import Path

if __package__ in (None, ""):  # run by path, e.g. %run in napari's console
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import load_config
from functions.imaging_masks import (
    available_masks,
    build_imaging_mask,
    describe_imaging_mask,
    parse_preset,
    remember_shown_mask,
)
from functions.mmcore_slm import MMCoreSLM, find_slm_core

# ---- parameters ----
PRESET = None    # e.g. "zoned_focus_3"; None -> must be given as main()'s argument or on the command line
ABERRATION_CORRECTION = False  # True -> add the stored aberration correction to PRESET
CHANNELS = None  # None -> imaging_masks.channels from the config (both channels)
BACKEND = None   # None (auto), "napari" or "micromanager"


AC_FLAGS = ("--aberration-correction", "-a")


def main(preset=None, aberration_correction=None):
    """
    Show `preset` (a mask name, or 'aberration_correction[_<mask>]') on
    the SLM; `aberration_correction=True` adds the stored correction to
    a plain mask name. Returns the mask's info dict.
    """
    if preset is None:
        args = sys.argv[1:]
        names = [a for a in args if a not in AC_FLAGS]
        preset = names[0] if names else PRESET
        if aberration_correction is None:
            aberration_correction = any(a in AC_FLAGS for a in args) or ABERRATION_CORRECTION

    config = load_config()
    masks = available_masks(config)
    if parse_preset(preset)[0] not in masks:
        raise SystemExit(
            f"Choose a phase mask: python -m experiments.load_phase_mask <mask> "
            f"[--aberration-correction] (got {preset!r}). Masks:\n"
            + "\n".join(f"  {name:<24} {m['label']}" for name, m in masks.items()))

    # fails before touching the SLM
    mask, info = build_imaging_mask(config, preset, CHANNELS,
                                    aberration_correction=bool(aberration_correction))
    core = find_slm_core(config, BACKEND)

    with MMCoreSLM(core, config) as slm:
        if slm.acquisition_running():
            raise RuntimeError("An acquisition is running - not changing the SLM mask mid-run.")
        slm.show_mask(mask)
    remember_shown_mask(mask, info)

    print(describe_imaging_mask(info))
    print("Shown on the SLM.")
    return info


if __name__ == "__main__":
    main()
