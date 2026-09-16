# ReflEx3D microscope controller — base code

Controls the microscope through a running Micro-Manager instance via
`pycromanager`, with an SLM addressed directly as a secondary
display.

## Layout

```
config/microscope_config.json   All hardware/software properties in one place
core/                            General infrastructure — never experiment-specific
  config.py                        Loads microscope_config.json
  session.py                       Creates the run folder for each experiment
  logging_setup.py                 One log file per run folder
  file_io.py                       Sole authority for disk writes (TIFF + metadata)
functions/                       General-purpose hardware control + orchestration
  laser.py                         Laser on/off/power
  camera.py                        Exposure + snap
  slm.py                           Phase mask generation + fullscreen display
  acquisition.py                   Orchestration routines (e.g. run_timelapse)
functions/zernike.py             Zernike polynomials (reused for AO later)
functions/phase_masks.py         Zernike probe patches, raster positions
functions/localization.py        Point-source centroid localization
experiments/                     Minimal scripts: parameters + function calls only
  example_timelapse.py             First example: flat-mask timelapse -> TIFF
  fourier_plane_alignment.py       Locates the pupil centre on the SLM
calibration/                     Infrequent, complex routines
  fourier_alignment.py             Fourier-plane alignment acquisition
  map_analysis.py                  Centre estimation from the d^2 map
  map_preview.py                   Preview figure builder
gui/                              (empty for now) tkinter UI, fully isolated
```

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
2. **Check `slm.monitor_index`.** This is the index into
   `screeninfo.get_monitors()` for the SLM's monitor — it depends on
   OS display ordering, not on Micro-Manager, so confirm it on the
   actual acquisition PC.
3. **Set `acquisition_defaults.root_folder`** to wherever you want
   run folders written.
4. `fourier_plane` is unused by the base timelapse experiment — it's
   there for the upcoming phase-mask-generation code (EDOF, Zernike
   terms, etc.) so those numbers live in one place from the start.

## Running the example experiment

With Micro-Manager open and connected to the hardware, and the SLM's
display active as a secondary monitor:

```
pip install -r requirements.txt
python -m experiments.example_timelapse
```

This displays a flat/neutral phase mask on the SLM, records a
50-frame timelapse (2 s interval, 100 ms exposure — edit the
constants at the top of the script to change this), and writes
`timelapse.tif` (TZCYX ImageJ hyperstack) plus a metadata JSON
sidecar into a new run folder under `acquisition_defaults.root_folder`.

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
