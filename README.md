# ReflEx3D microscope controller — base code

Controls the microscope through a running Micro-Manager instance.

The codebase is migrating from `pycromanager` to
[`pymmcore-plus`](https://github.com/pymmcore-plus/pymmcore-plus) /
[`napari-micromanager`](https://github.com/pymmcore-plus/napari-micromanager) /
[`pymmcore-widgets`](https://github.com/pymmcore-plus/pymmcore-widgets), for
better performance and to enable real-time image processing during
acquisition. The `pycromanager` code isn't removed by this migration —
it's kept as **legacy** (see "Legacy vs. pymmcore-plus" below) while
new experiments move to the pymmcore-plus stack. The legacy path still
addresses the SLM directly as a secondary display; the pymmcore-plus
path addresses it as a proper Micro-Manager device instead (the
"Generic SLM: Spatial light modulator controlled through computer
graphics" device), through `CMMCorePlus`.

## Layout

```
config/microscope_config.json   All hardware/software properties in one place
core/                            General infrastructure — never experiment-specific
  config.py                        Loads microscope_config.json
  session.py                       Creates the run folder for each experiment
  logging_setup.py                 One log file per run folder
  file_io.py                       Sole authority for disk writes (TIFF + metadata)
  mmcore.py                        Builds a CMMCorePlus instance from the
                                    "pymmcore_plus" config section
functions/                       General-purpose hardware control + orchestration
  laser.py                         (legacy, pycromanager) Laser on/off/power
  mmcore_laser.py                  pymmcore-plus equivalent of laser.py
  camera.py                        (legacy, pycromanager) Exposure + snap
  slm.py                           Phase mask generation (hardware-agnostic,
                                    shared) + (legacy) secondary-display window
  mmcore_slm.py                    pymmcore-plus equivalent of SLMDisplay:
                                    addresses the SLM as a Micro-Manager device
  acquisition.py                   Orchestration routines: run_timelapse (legacy)
                                    and build_timelapse_sequence/run_timelapse_mda
                                    (pymmcore-plus MDA)
  napari_preview.py                Live napari preview layer for MDA runs
functions/zernike.py             Zernike polynomials (reused for AO later)
functions/phase_masks.py         Zernike probe patches, raster positions
functions/localization.py        Point-source centroid localization
experiments/                     Minimal scripts: parameters + function calls only
  example_timelapse.py             (legacy, pycromanager) flat-mask timelapse -> TIFF
  example_timelapse_mda.py         Same timelapse via pymmcore-plus's MDA engine,
                                    with a live napari preview
  fourier_plane_alignment.py       Locates the pupil centre on the SLM
calibration/                     Infrequent, complex routines
  fourier_alignment.py             Fourier-plane alignment acquisition
  map_analysis.py                  Centre estimation from the d^2 map
  map_preview.py                   Preview figure builder
gui/                              (empty for now) tkinter UI, fully isolated
```

## Legacy vs. pymmcore-plus

`experiments/example_timelapse.py` and the modules it calls
(`functions/laser.py`, `functions/camera.py`,
`functions/acquisition.py`'s `run_timelapse`) go through `pycromanager`'s
`Core`, which proxies a running Micro-Manager GUI instance. They're
unchanged by this migration and stay as the "legacy" path.

`experiments/example_timelapse_mda.py` and its equivalents
(`core/mmcore.py`, `functions/mmcore_laser.py`, `functions/mmcore_slm.py`,
`functions/acquisition.py`'s `build_timelapse_sequence`/`run_timelapse_mda`,
`functions/napari_preview.py`) instead build a `CMMCorePlus` instance
directly (no separate Micro-Manager GUI process needed), run the
acquisition through pymmcore-plus's MDA engine, and stream frames live
into a napari viewer. New experiments should generally use this path.
`functions/slm.py`'s `flat_mask` (pure numpy) is shared by both stacks,
but its `SLMDisplay` (a plain secondary-display window) is legacy-only:
the pymmcore-plus path uses `functions/mmcore_slm.py`'s `MMCoreSLM`
instead, which addresses the SLM as a Micro-Manager device through
`mmc` rather than opening its own window.

The rule that keeps this maintainable: `experiments/*.py` files
should never contain hardware-control logic directly — only
parameters and calls into `functions/`. New experiments are new,
short files in `experiments/`; the general-use code underneath
doesn't change.

## Before running anything

1. **Fill in `config/microscope_config.json`.** In particular,
   `stage.xy_device_label`, `stage.z_device_label`, and
   `micromanipulator.device_label` are placeholders
   (`REPLACE_WITH_MM_DEVICE_NAME`) — swap in the exact device labels
   Micro-Manager uses for your hardware config. Double-check
   `laser.device_label` and `camera.device_label` against your
   Micro-Manager config too — they're filled in with the SPIM2by2
   defaults but device labels can vary between MM config files.
2. **For the legacy path**, check `slm.monitor_index`. This is the
   index into `screeninfo.get_monitors()` for the SLM's monitor — it
   depends on OS display ordering, not on Micro-Manager, so confirm it
   on the actual acquisition PC.
3. **Set `acquisition_defaults.root_folder`** to wherever you want
   run folders written.
4. `fourier_plane` is unused by the base timelapse experiment — it's
   there for the upcoming phase-mask-generation code (EDOF, Zernike
   terms, etc.) so those numbers live in one place from the start.
5. **For the pymmcore-plus path only**, fill in `pymmcore_plus.device_adapter_path`
   (folder containing your Micro-Manager device adapter DLLs/.so files) and
   `pymmcore_plus.system_config_path` (the `.cfg` hardware configuration
   file) — these replace the running Micro-Manager GUI instance that the
   legacy `pycromanager` path connects to. Also add a "Generic SLM:
   Spatial light modulator controlled through computer graphics"
   device to that MM hardware config, positioned as a fullscreen
   window on the SLM's monitor, and check `slm.device_label` matches
   its device label exactly. `slm.resolution` must match that
   device's resolution too — `MMCoreSLM` checks this at runtime and
   raises an error on a mismatch.

## Running the example experiment

**Legacy (pycromanager):** with Micro-Manager open and connected to the
hardware, and the SLM's display active as a secondary monitor:

```
pip install -r requirements.txt
python -m experiments.example_timelapse
```

This displays a flat/neutral phase mask on the SLM, records a
50-frame timelapse (2 s interval, 100 ms exposure — edit the
constants at the top of the script to change this), and writes
`timelapse.tif` (TZCYX ImageJ hyperstack) plus a metadata JSON
sidecar into a new run folder under `acquisition_defaults.root_folder`.

**pymmcore-plus, with a live napari preview:** no separate
Micro-Manager GUI process needed — `core/mmcore.py` builds the core
directly from `pymmcore_plus.device_adapter_path`/`system_config_path`,
with the Generic SLM device (`slm.device_label`) loaded as part of
that system config:

```
pip install -r requirements.txt
python -m experiments.example_timelapse_mda
```

This runs the same flat-mask, 50-frame timelapse, but through
pymmcore-plus's MDA engine (`mmc.run_mda`) instead of a manual snap
loop, and opens a napari viewer that fills in live as each frame is
acquired. Close the viewer window to end the run.

## Adding a new experiment

1. Add any new general-purpose hardware calls to `functions/`.
2. Add any new orchestration sequences to `functions/acquisition.py`
   (or a new module there for a distinct instrument mode).
3. Write a new short script in `experiments/`, following
   `example_timelapse.py`'s shape: load config, open a `Session`,
   set up logging, run the sequence, save.


## Fourier-plane alignment

`experiments/fourier_plane_alignment.py` finds the pupil centre on the
SLM for one polarization channel. A circular Zernike probe patch is
rastered over the SLM; at each position it is displayed with
+amplitude (0 -> 2*pi) and then -amplitude (2*pi -> 0), the bead is
localized in both frames, and the squared separation d^2 is recorded.

* **tilt** -> d^2 is a **maximum** at the centre (max pupil coverage)
* **defocus** -> d^2 is a **minimum** at the centre (no net tilt when
  centred), sitting inside a ring
* either way d^2 -> 0 where the patch misses the pupil entirely

Before running, select a ROI in Micro-Manager showing only one
polarization channel with an isolated microsphere in it. The script
asks which channel it is and which probe mode to use, shows you a
verification frame to confirm the bead is localizable, rasters, then
shows a preview of the map with the estimated centre. Only if you
accept it does it write `fourier_plane.channels.<channel>.center_px`
back into the config (backing the old config up first).

### Practical notes

* **Amplitude is mode-dependent.** Tilt tolerates a full 2*pi
  peak-to-valley across the patch; defocus works best near *pi*,
  because a 2*pi defocus stroke scatters light so widely that the
  centroid stops tracking it. Both defaults are in the config and
  should be re-tuned on the real instrument.
* **Tilt is the sharper probe.** In simulation against a known ground
  truth, tilt recovered the centre to ~0.5 px and defocus to ~7 px:
  tilt gives one clean peak, while defocus's central dip is shallow
  and sits inside a ring. Use tilt as the primary and defocus as an
  independent cross-check.
* **Patch size matters.** `patch_diameter_px` must be a fraction of
  the pupil diameter (its radius is a good start) so the patch can
  *partially* overlap the pupil — that partial overlap is the signal.
  A patch much larger than the pupil gives no contrast.
* **Localization method.** `frame_centroid` is the default and is the
  theoretically correct estimator here: the PSF intensity centroid
  equals the pupil-area-weighted mean phase gradient, which is exactly
  why tilt scales with coverage. It assumes one isolated bead in the
  ROI. `windowed` is more robust to stray light but tracks the
  dominant lobe (produced by the *unmodulated* pupil) and so shows
  weaker contrast.
* **`grey_level_2pi` must be right.** It comes from the SLM's phase
  LUT. If it's wrong, every phase mask in the codebase is miscalibrated.
* **Time.** Cost is `2 x n_positions x (settle + exposure)`, with
  settle = `settle_frames / refresh_rate_hz` = 100 ms at 30 Hz. The
  script prints an estimate and asks before starting. Start coarse
  (`step_px` 60-80), then re-run with `X_RANGE`/`Y_RANGE` around the
  coarse result for a fine pass.
* The raw map (`.npz` + `.csv`) is saved immediately after the raster,
  *before* you approve anything, so a long acquisition is never lost
  to a rejected fit. It can be re-analysed without re-acquiring.
