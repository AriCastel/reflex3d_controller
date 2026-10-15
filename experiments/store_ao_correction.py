"""
experiments/store_ao_correction.py

Store the coefficients of an earlier aberration-correction run in
config/phase_masks.json (under its method and channel), where the
imaging mask scripts and the timelapse read them - for a run that
wasn't stored when it finished (or one that predates the store).

Run with the run folder (or its ao_correction_metadata.json):
    python -m experiments.store_ao_correction <run folder>
"""
import sys

from core.config import load_config
from core.file_io import load_ao_metadata, write_phase_mask_coefficients
from calibration.fourier_alignment import prompt_yes_no
from functions.adaptive_optics import AO_METHOD, coefficient_table, phase_mask_entry

# ---- parameters ----
RUN = None  # run folder; None -> first command-line argument


def main(run=None):
    run = run or RUN or (sys.argv[1] if len(sys.argv) > 1 else None)
    if run is None:
        sys.exit("Usage: python -m experiments.store_ao_correction <run folder>")
    config = load_config()
    metadata, run_folder = load_ao_metadata(run)
    method = metadata.get("method", AO_METHOD)  # runs before the store were all SPSA
    channel = metadata["channel"]
    coefficients = metadata["coefficients_rad_rms"]

    print(f"Run {run_folder}: channel '{channel}', method '{method}', "
          f"measured {metadata.get('finished_at')}")
    print(coefficient_table([int(j) for j in coefficients], list(coefficients.values())))
    store = config.phase_masks_path
    if not prompt_yes_no(f"Store in {store.name} as the '{method}' correction for channel "
                         f"'{channel}' (replacing any stored one)?", default=True):
        return
    _, backup = write_phase_mask_coefficients(
        store, method, channel, phase_mask_entry(metadata, run_folder))
    print(f"Stored in {store}" + (f" (previous file backed up to {backup.name})" if backup else ""))


if __name__ == "__main__":
    main()
