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
config/phase_masks.json         Stored phase-mask coefficients (AO corrections),
                                 by method then channel
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
  mmcore_micromanipulator.py       Sutter MPC stage in camera axes (config axis map),
                                    backlash-safe one-direction scans
  mmcore_camera.py                 pymmcore-plus per-channel camera ROIs
                                    (camera.channel_rois in the config)
  slm.py                           (legacy) SLM as a secondary-display window
  mmcore_slm.py                    The SLM as Micro-Manager's Generic SLM device, through
                                    pymmcore-plus or pycromanager (Micro-Manager 2.0)
  acquisition.py                   Orchestration routines: run_timelapse (legacy)
                                    and build_timelapse_sequence/run_timelapse_mda
                                    (pymmcore-plus MDA)
  napari_preview.py                napari viewers: live MDA preview, live
                                    full-frame preview with channel ROI boxes,
                                    verification frame
functions/zernike.py             Zernike polynomials (incl. Noll-indexed, RMS-normalized)
functions/psf_metrics.py         Guide-star PSF metrics for AO (pluggable registry)
functions/adaptive_optics.py     Pupil Zernike masks, guide-star camera, SPSA optimizer,
                                  results figure
functions/imaging_masks.py       Imaging phase masks per channel (stored correction,
                                  EDOF, both, flat), running-GUI core lookup
functions/phase_masks.py         Flat masks, Zernike probe patches, raster positions
functions/localization.py        Point-source centroid localization
experiments/                     Minimal scripts: parameters + function calls only
  example_timelapse.py             (legacy, pycromanager) flat-mask timelapse -> TIFF
  example_timelapse_mda.py         Same timelapse via pymmcore-plus's MDA engine,
                                    with a live napari preview
  fourier_plane_alignment.py       Locates the pupil centre on the SLM (pymmcore-plus)
  fourier_plane_refinement.py      Refines that centre with a fine raster around it
  aberration_correction.py         Sensorless AO on a guide star (pymmcore-plus)
  store_ao_correction.py           Store an earlier AO run's coefficients in phase_masks.json
  psf_characterization.py          Depth stack of a guide star with a chosen phase mask
  imaging_gui.py                   napari-micromanager GUI + 'SLM phase mask' panel
  load_phase_mask.py               Load a mask preset onto the SLM (stored correction,
                                    EDOF, both, flat), from the napari GUI or with
                                    Micro-Manager 2.0
calibration/                     Infrequent, complex routines
  fourier_alignment.py             Fourier-plane alignment acquisition
  fourier_refinement.py            Helpers for the refinement pass (stored centres)
  map_analysis.py                  Centre estimation from the d^2 map
  map_preview.py                   Preview figure builder
gui/                              (empty for now) tkinter UI, fully isolated
tests/
  e2e_test.py                      Alignment acquisition -> analysis -> config
                                    write-back on a simulated pymmcore-plus
                                    microscope (python tests/e2e_test.py)
  refinement_e2e_test.py           Refinement of an offset centre on the same simulated
                                    microscope (python tests/refinement_e2e_test.py)
  sim_validate.py                  Centre recovery on a simulated pupil
  ao_sim_test.py                   Aberration correction on a simulated microscope
                                    with a known aberration (python tests/ao_sim_test.py)
  mm2_slm_test.py                  load_phase_mask with Micro-Manager 2.0, against a
                                    fake pycromanager Core (python tests/mm2_slm_test.py)
  psf_characterization_test.py     PSF depth stack on a simulated microscope whose stage
                                    has backlash (python tests/psf_characterization_test.py)
  imaging_masks_test.py            phase_masks.json store, imaging masks, mask scripts and
                                    the timelapse's PHASE_MASK on a simulated microscope
                                    (xvfb-run -a python tests/imaging_masks_test.py)
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
`functions/slm.py`'s `SLMDisplay` (a plain secondary-display window) is
legacy-only: the pymmcore-plus path uses `functions/mmcore_slm.py`'s
`MMCoreSLM` instead, which addresses the SLM as a Micro-Manager device
through `mmc` rather than opening its own window. `flat_mask` lives in
`functions/phase_masks.py` and is shared by both.

`experiments/fourier_plane_alignment.py` has been moved to the
pymmcore-plus path outright (no legacy copy). Without the
Micro-Manager GUI there's nowhere to draw the camera ROI by hand, so
each channel's ROI is read from `camera.channel_rois` in the config
and applied in code (`functions/mmcore_camera.py`).

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
   **Set `camera.channel_rois`** (a square ROI per polarization
   channel: `position_px` = top-left `[x, y]`, `size_px` = side, in
   full-frame camera pixels) before running the Fourier-plane
   alignment. They only need setting once, since each channel always
   lands on the same part of the sensor; the defaults are placeholders.
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

The script runs on the pymmcore-plus stack, so no Micro-Manager GUI
is needed, and it sets the camera ROI itself:

1. A napari window streams the **full camera frame** live, with a flat
   mask on the SLM and the laser at the alignment power, and each
   channel's ROI from `camera.channel_rois` drawn as a labelled box.
   Position the sample so an isolated microsphere sits inside the ROI
   of the channel you want to align, then **close the window**.
2. In the terminal, choose the channel and the probe mode, and confirm
   the estimated acquisition time.
3. The camera is cropped to that channel's ROI and a verification frame
   is shown in napari with the localized bead circled. Close it and
   confirm that the bead is correctly localized.
4. The raster runs, then a preview of the map with the estimated centre
   is shown. Only if you accept it does the script write
   `fourier_plane.channels.<channel>.center_px` back into the config
   (backing the old config up first).

The camera ROI is cleared back to full frame when the script ends,
including when it's aborted.

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


### Refining the centre (`fourier_plane_refinement.py`)

Rastering the whole SLM gets expensive as the step shrinks (time goes
as 1/step^2). Once `fourier_plane.channels.<channel>.center_px` holds a
coarse estimate, `experiments/fourier_plane_refinement.py` rasters only
a **disk around it** (default radius: `fourier_plane.radius_px`) at a
**finer step** (default `fourier_alignment.step_px / 4`), with the same
probe patch and d^2 measurement. With the shipped numbers that is ~650
positions at 25 px instead of ~3500 for the whole SLM at the same step.

It follows the same flow as the coarse experiment - live ROI preview,
channel choice (only channels with a stored centre are offered; the
probe mode defaults to the one used for the stored centre), time
estimate, point-source verification, then the raster with a `tqdm`
progress bar - and saves the raw map (`.npz` + `.csv`) before asking
anything. The preview PNG shows the rastered area on the SLM, the fine
map with the previous and refined centres, and the radial profile. The
config is only updated (with a `.json.bak` backup) after you confirm.

Because the window only covers part of the response, the support
centroid used by the coarse analysis does not apply. The refined centre
is the point about which the map is best described as a function of
radius alone (`refine_fourier_center` in `calibration/map_analysis.py`).
The result is flagged if that radial model explains < 50% of the map's
variance, if the centre lands at the edge of the search area (half the
window radius) or if many positions failed to localize. Tune
`REFINE_FACTOR`, `STEP_PX` and `WINDOW_RADIUS_PX` at the top of the
script; the window must reach into the fall-off of the response (for
tilt, beyond `pupil radius - patch radius` from the centre).


## Aberration correction (sensorless adaptive optics)

`experiments/aberration_correction.py` corrects one channel's aberrations
on a guide star (an isolated microsphere). The SLM phase over that
channel's pupil is a sum of Zernike modes (Noll order, RMS-normalized,
so coefficients are radians RMS), and the coefficients are optimized by
**SPSA**, a stochastic gradient descent: each iteration probes one random
+/-1 direction in coefficient space with two snaps, F(a + c*delta) and
F(a - c*delta), whatever the number of modes. Settings live in the
`adaptive_optics` config section.

Flow: the live ROI preview (as in the alignment) -> choose the channel
(its Fourier-plane centre must already be calibrated) -> confirm the
guide star -> the camera is cropped to a small square around it, a dark
frame is taken -> optimize -> the flat and corrected masks are snapped
back to back. The run folder gets `ao_correction.png` (metric vs
iteration, original vs corrected PSF, final phase mask, coefficients),
`ao_correction.npz` and `ao_correction_metadata.json` (coefficients by
Noll index), and the coefficients are printed.

* **Metrics are pluggable.** They live in `functions/psf_metrics.py`; lower
  is always better. Write a small class with `__call__(image)` (and
  optionally `set_reference(image)`, called with the uncorrected PSF),
  add it to `METRICS`, and select it by name in
  `adaptive_optics.metric`. The default, `power_weighted_second_moment`,
  is the previous implementation's second moment multiplied by
  (reference power / current power), so losing light is penalized;
  `encircled_energy` and `sharpness` are also power-aware.
  `second_moment` (power-blind) is kept for comparison.
* **Photon budget.** Two snaps per iteration (the metric at the current
  point is the mean of the two), a small guide-star ROI for fast
  readout, a precomputed Zernike basis, and early stopping
  (`patience`). Each iteration costs about 2 x (SLM settle + exposure),
  ~0.3 s with the default 100 ms settle and 20 ms exposure.
* **Tuning.** In simulation, `gain` 0.1 converged in ~30-50 iterations;
  0.5 overshot badly first and took ~3x longer. `blank_laser_during_settle`
  switches the laser off while the SLM settles, cutting illumination
  ~in half, if the laser switches cleanly and fast.
* **Storing the result.** After the figure is closed, the script asks
  whether to store the coefficients in `config/phase_masks.json`, under
  the method (`spsa`) and the channel, replacing what was stored there
  (the previous file is kept as `phase_masks.json.bak`). For a run that
  wasn't stored, `python -m experiments.store_ao_correction <run folder>`
  does the same from its `ao_correction_metadata.json`. Other AO
  methods will get their own key next to `spsa`.


## Imaging with phase masks

Each channel's pupil on the SLM gets a sum of Zernike modes (same
convention as the AO: Noll order, rad RMS), from the stored aberration
correction and/or the extended-depth-of-field (EDOF) terms in the
`imaging_masks` config section (primary spherical, Noll 11).
`experiments/load_phase_mask.py` loads one of four presets onto both
channels (`imaging_masks.channels`), given as its argument:

| preset | mask |
| --- | --- |
| `aberration_correction` | stored correction (`imaging_masks.aberration_correction_method`, `spsa`) |
| `edof` | primary spherical only (`imaging_masks.edof_zernike_rad_rms`) |
| `aberration_correction_edof` | both, added term by term (the correction's own Noll 11 plus the EDOF amount) |
| `flat` | completely flat |

The same script works with the napari GUI and with Micro-Manager 2.0:
it uses the napari GUI's core if it runs inside the GUI's process, and
otherwise connects to a running Micro-Manager 2.0 through pycromanager
(`BACKEND` at the top forces one or the other).

**From the napari GUI.** Micro-Manager hardware can only be open in one
process, and the SLM only shows a mask while the process that loaded it
is alive, so the script runs *inside* the GUI's process:

1. `python -m experiments.imaging_gui` opens napari with
   napari-micromanager on the hardware config in `pymmcore_plus` and a
   flat mask on the SLM.
2. Position the sample and set exposure, laser power etc. with the
   napari-micromanager widgets.
3. Load the mask with a button in the **SLM phase mask** panel, or run
   the script from napari's console (`%run experiments/load_phase_mask.py edof`;
   with your own napari-micromanager window, give the full path). Each
   load writes a small run folder with the mask (`slm_mask.tif`) and
   what it is made of (`slm_mask_metadata.json`: preset, coefficients
   per channel, which stored correction), so data can be matched to the
   mask that was on the SLM. A missing stored correction is reported
   and the SLM is left as it was; masks aren't changed while an
   acquisition is running.
4. Start the timelapse from napari-micromanager's MDA widget. The mask
   stays up until another one is loaded or the GUI is closed.

**From Micro-Manager 2.0.** When the microscope is run from a
Micro-Manager 2.0 instance instead, run it from a terminal while
Micro-Manager is open:

```
python -m experiments.load_phase_mask aberration_correction
python -m experiments.load_phase_mask edof
python -m experiments.load_phase_mask aberration_correction_edof
python -m experiments.load_phase_mask flat
```

The mask goes through Micro-Manager's Generic SLM device
(`slm.device_label`) via pycromanager, so it stays up after the script
exits, until another is loaded. Micro-Manager
must have the pycro-manager server enabled (Tools > Options > "Run
pycro-manager server on port 4827") and the Generic SLM device in its
hardware config, at `slm.resolution`. Each load is recorded in a run
folder as above.

**Headless timelapse.** `experiments/example_timelapse_mda.py` takes
`PHASE_MASK` (one of the four presets, default `"flat"`) and
`PHASE_MASK_CHANNELS` at the top. The mask is built before any hardware
is touched (so a missing correction stops the run before it starts),
shown for the whole acquisition, and saved into the run folder.

The EDOF amplitude (`edof_zernike_rad_rms`, default 2.0 rad RMS on
Noll 11, ~1 wave peak-to-valley) is a starting point to tune on the
instrument; its sign sets which side of focus the depth of field
extends towards. Being an RMS-normalized Zernike term, it has no
defocus component, so the best focus stays put.


## 3D PSF characterization

`experiments/psf_characterization.py` records a depth (Z) stack of a
guide star with any of the imaging phase masks, by moving the
micromanipulator the sample is mounted on. Thanks to **Luis Bastidas**,
who first worked out how to drive this micromanipulator and wrote the
original Z-stack script this is based on.

Flow: the live ROI preview -> choose the channel -> confirm the guide
star in the verification frame -> the camera is cropped to a square
around it (`guide_star_roi_px`) -> choose the phase mask (Enter = flat)
-> confirm the scan -> the stack -> `dark_frames` (200) laser-off frames
with the same ROI, whose mean is subtracted from every plane -> the stage
returns to its starting depth. The run folder gets `psf_zstack.tif`
(float32, dark-subtracted, ImageJ ZYX with the Z spacing, top of the range
first), `psf_zstack_metadata.json` (commanded and read-back depth of every
plane, settings, mask), `dark_mean.tif` and the mask. Defaults are in the
`psf_characterization` config section (+/-3 um in 0.125 um steps).

Things the micromanipulator code (`functions/mmcore_micromanipulator.py`)
takes care of:

* **Camera axes, not stage axes.** The stage is sometimes re-mounted, so
  'Current Z' isn't necessarily the objective-sample distance.
  `micromanipulator.axis_properties` maps the camera's `horizontal`,
  `vertical` and `depth` axes to the stage properties; the scan moves
  whatever `depth` maps to. **Check it whenever the stage is re-mounted.**
  The shipped `depth = "Current X"` is what the previous setup's script
  scanned; horizontal/vertical are placeholders.
* **Positions are set through 'Current X/Y/Z'**, the most reliable way to
  move this stage.
* **Backlash.** The microstep is 62.5 nm, so the step must be a whole
  number of microsteps (checked before anything moves), but a position
  is only reproducible from one direction. The stage first goes
  `overshoot_steps` steps *above* the top of the range, comes back down
  to it, and then only moves down. The return to the start position is
  approached from above in the same way.
* If anything fails mid-scan, the laser is switched off, the stage goes
  back to its starting depth and the camera ROI is cleared.

