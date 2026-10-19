"""
experiments/imaging_gui.py

Opens the napari GUI for imaging: napari-micromanager on the hardware
config from config/microscope_config.json (pymmcore_plus section), with
a flat mask on the SLM, plus two panels:

* 'SLM phase mask': buttons that run experiments/load_phase_mask.py with
  each preset inside this process;
* 'Timelapse': name, exposure (ms), laser power (mW) and number of frames
  (with the resulting sampling rate), which runs experiments/timelapse.py
  - a continuous timelapse at the current position, saved under the
  name in the usual dated run folder.

Workflow:
1. Position the sample and set exposure, laser power etc. with the
   napari-micromanager widgets (live view, property browser).
2. Load the phase mask for the acquisition with a button in the
   'SLM phase mask' panel (or `%run experiments/load_phase_mask.py <preset>`
   in napari's console - same thing). Each load is recorded in its own
   small run folder (mask TIFF + what it's made of).
3. Fill in the 'Timelapse' panel and press Start (Stop ends it early,
   keeping the frames so far). The frames show up in the viewer as they
   come in, and the panel reports where they were saved and the measured
   sampling rate. The mask stays on the SLM until another one is loaded
   or the GUI is closed. (napari-micromanager's own MDA widget is still
   there for anything else, but don't give it XYZ positions: this stage
   must only be moved through its 'Current X/Y/Z' properties.)

Run with:
    python -m experiments.imaging_gui
"""
import warnings

import napari

from core.config import load_config
from core.mmcore import load_mmcore
from experiments import load_phase_mask, timelapse
from functions.mmcore_laser import laser_off_mmcore
from functions.mmcore_slm import MMCoreSLM
from functions.imaging_masks import build_imaging_mask, remember_shown_mask
from functions.napari_preview import add_script_panel, add_timelapse_panel

# ---- parameters: panel button label -> load_phase_mask preset ----
MASK_PRESETS = {
    "Aberration correction": "aberration_correction",
    "EDOF (primary spherical)": "edof",
    "Aberration correction + EDOF": "aberration_correction_edof",
    "Flat": "flat",
}


def main():
    config = load_config()
    # The napari window (and so the Qt app) must exist BEFORE the core is
    # created: pymmcore-plus only gives the core thread-safe Qt signals if a
    # Qt app is running then. Otherwise the widgets get updated straight from
    # the acquisition thread during a timelapse, which crashes Qt.
    viewer = napari.Viewer(title="ReflEx3D")
    mmc = load_mmcore(config)
    if type(mmc.events).__name__ != "QCoreSignaler":
        warnings.warn("The Micro-Manager core was created before the GUI, so its events "
                      "aren't thread-safe for the GUI: restart with only "
                      "`python -m experiments.imaging_gui`.", stacklevel=1)
    flat, flat_info = build_imaging_mask(config, "flat")
    with MMCoreSLM(mmc, config) as slm:
        slm.show_mask(flat)
    remember_shown_mask(flat, flat_info)

    from napari_micromanager import MainWindow

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

    def start_timelapse(name, exposure_ms, laser_power_mW, num_frames, done):
        def finished(summary):
            rate = summary["measured_sampling_rate_hz"]
            done(f"{'Done' if summary['completed'] else 'Stopped'}: "
                 f"{summary['frames_acquired']}/{num_frames} frames saved to {summary['path']}"
                 + (f" (measured {rate:.2f} Hz)" if rate else ""))
        timelapse.main(name, num_frames, exposure_ms, laser_power_mW, mmc=mmc,
                       on_finished=finished)

    defaults = config.get("acquisition_defaults", "timelapse", default={})
    add_timelapse_panel(
        viewer, start_timelapse, stop=mmc.mda.cancel,
        exposure_ms=defaults.get("exposure_ms", 50),
        laser_power_mW=config.get("laser", "default_power_mW", default=0.0),
        num_frames=defaults.get("num_frames", 100),
        max_laser_mW=config.get("laser", "max_power_mW"),
    )

    try:
        napari.run()
    finally:
        laser_off_mmcore(mmc, config)


if __name__ == "__main__":
    main()
