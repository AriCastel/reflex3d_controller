"""
experiments/imaging_gui.py

Opens the napari GUI for imaging: napari-micromanager on the hardware
config from config/microscope_config.json (pymmcore_plus section), with
a flat mask on the SLM, plus an 'SLM phase mask' panel whose buttons
run experiments/load_phase_mask.py with each preset inside this process.

Workflow:
1. Position the sample and set exposure, laser power etc. with the
   napari-micromanager widgets (live view, property browser).
2. Load the phase mask for the acquisition with a button in the
   'SLM phase mask' panel (or `%run experiments/load_phase_mask.py <preset>`
   in napari's console - same thing). Each load is recorded in its own
   small run folder (mask TIFF + what it's made of).
3. Start the timelapse from napari-micromanager's MDA widget. The mask
   stays on the SLM until another one is loaded or the GUI is closed.

Run with:
    python -m experiments.imaging_gui
"""
import napari

from core.config import load_config
from core.mmcore import load_mmcore
from experiments import load_phase_mask
from functions.mmcore_laser import laser_off_mmcore
from functions.mmcore_slm import MMCoreSLM
from functions.napari_preview import add_script_panel
from functions.phase_masks import flat_mask

# ---- parameters: panel button label -> load_phase_mask preset ----
MASK_PRESETS = {
    "Aberration correction": "aberration_correction",
    "EDOF (primary spherical)": "edof",
    "Aberration correction + EDOF": "aberration_correction_edof",
    "Flat": "flat",
}


def main():
    config = load_config()
    mmc = load_mmcore(config)
    with MMCoreSLM(mmc, config) as slm:
        slm.show_mask(flat_mask(config))

    from napari_micromanager import MainWindow

    viewer = napari.Viewer(title="ReflEx3D")
    try:
        window = MainWindow(viewer, mmcore=mmc)
    except TypeError:
        # Older napari-micromanager versions take no `mmcore` and always
        # drive CMMCorePlus.instance() - the core load_mmcore returned.
        window = MainWindow(viewer)
    viewer.window.add_dock_widget(window, name="MicroManager", area="top")
    add_script_panel(viewer, "SLM phase mask", {
        label: (lambda preset=preset: load_phase_mask.main(preset))
        for label, preset in MASK_PRESETS.items()
    })

    try:
        napari.run()
    finally:
        laser_off_mmcore(mmc, config)


if __name__ == "__main__":
    main()
