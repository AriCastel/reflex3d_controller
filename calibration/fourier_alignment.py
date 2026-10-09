"""
calibration/fourier_alignment.py

Acquisition side of the Fourier-plane alignment procedure.

Procedure
---------
0. Show a live full-frame preview with every channel's camera ROI
   outlined (flat mask on the SLM), so the sample can be positioned.
1. Verify a point source (microsphere) is visible and localizable
   inside the chosen channel's ROI, with a flat mask on the SLM.
2. Raster a circular Zernike probe patch across the whole SLM. At each
   position, display the patch with +amplitude, snap; display it with
   -amplitude, snap. Localize the point source in both frames and
   record the squared separation d^2.
3. The resulting map is analysed by calibration/map_analysis.py.

Sign convention: +amplitude ramps 0 -> 2*pi across the patch,
-amplitude ramps 2*pi -> 0. The PSF shift reverses with the sign, so
d^2 measures how strongly the patch is acting as a tilt on the pupil.

Hardware goes through pymmcore-plus: `mmc` is a CMMCorePlus instance
and `slm` a functions/mmcore_slm.py MMCoreSLM. The raster is a plain
mask -> settle -> snap loop rather than an MDA sequence, because the
MDA engine snaps straight after displaying an SLM image (no settle
time) and only logs a warning if setting the image fails.

Nothing here writes to disk — frames, maps and metadata are returned
to the caller, which routes all saving through core/file_io.py.
"""
import time
from datetime import datetime

import numpy as np

from functions.localization import localize_psf, squared_separation
from functions.mmcore_camera import get_all_channel_rois
from functions.mmcore_laser import laser_off_mmcore, laser_on_mmcore, set_laser_power_mmcore
from functions.napari_preview import run_live_roi_preview
from functions.phase_masks import flat_mask, raster_positions, zernike_patch_mask
from functions.zernike import MODE_INDICES


def slm_settle_seconds(config):
    """
    How long to wait after pushing a mask before snapping.

    The SLM runs at a fixed refresh rate (30 Hz here), so a frame
    snapped too early shows the PREVIOUS mask. Waiting a few frame
    periods covers both the refresh latency and liquid-crystal
    relaxation.
    """
    rate = config.get("slm", "refresh_rate_hz", default=30)
    frames = config.get("slm", "settle_frames", default=3)
    return float(frames) / float(rate)


def estimate_duration_s(config, n_positions, exposure_ms, overhead_ms=25.0):
    """Rough wall-clock estimate: 2 masks + 2 snaps per position."""
    settle = slm_settle_seconds(config)
    per_frame = settle + exposure_ms / 1000.0 + overhead_ms / 1000.0
    return n_positions * 2.0 * per_frame


def format_duration(seconds):
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h} h {m} min"
    if m:
        return f"{m} min {s} s"
    return f"{s} s"


def preview_sample_positioning(mmc, config, slm, logger, exposure_ms,
                               laser_power_mW):
    """
    Live full-frame preview with every channel's ROI outlined, so the
    scientist can check the sample is positioned before acquisition.
    A flat mask is displayed first - with a probe or leftover mask on
    the SLM the preview wouldn't show the unaberrated sample. Blocks
    until the preview window is closed.
    """
    rois = get_all_channel_rois(config)
    slm.show_mask(flat_mask(config))
    time.sleep(slm_settle_seconds(config))

    mmc.setExposure(exposure_ms)
    set_laser_power_mmcore(mmc, config, laser_power_mW)
    laser_on_mmcore(mmc, config)
    logger.info(
        "Live ROI preview open (flat mask on the SLM). Close the window "
        "once the sample is positioned."
    )
    try:
        run_live_roi_preview(
            mmc, rois,
            title="ReflEx3D — position the sample, then close this window",
        )
    finally:
        laser_off_mmcore(mmc, config)
    logger.info("Live ROI preview closed.")


def verify_point_source(mmc, config, slm, logger, exposure_ms,
                        laser_power_mW, localization_method):
    """
    Snap one frame with a flat mask so the scientist can confirm the
    point source is visible and correctly localized before committing
    to a long raster. The camera ROI must already be set.

    Returns (frame, localization_or_None).
    """
    slm.show_mask(flat_mask(config))
    time.sleep(slm_settle_seconds(config))

    mmc.setExposure(exposure_ms)
    set_laser_power_mmcore(mmc, config, laser_power_mW)
    laser_on_mmcore(mmc, config)
    try:
        frame = mmc.snap()
    finally:
        laser_off_mmcore(mmc, config)

    loc = localize_psf(frame, method=localization_method)
    if loc is None:
        logger.warning(
            "No point source passed the SNR test in the verification frame."
        )
    else:
        logger.info(
            f"Verification: source at ({loc['x']:.1f}, {loc['y']:.1f}) "
            f"px, peak {loc['peak']:.0f}, SNR {loc['snr']:.1f}"
        )
    return frame, loc


def acquire_d2_map(
    mmc,
    config,
    slm,
    logger,
    mode,
    amplitude_rad,
    patch_diameter_px,
    step_px,
    exposure_ms,
    laser_power_mW,
    localization_method="frame_centroid",
    x_range=None,
    y_range=None,
    progress_every=25,
    positions=None,
    position_mask=None,
    use_tqdm=False,
):
    """
    Raster the probe patch over the SLM and build the d^2 map.

    By default the whole SLM is rastered on a `step_px` grid. A caller
    that already knows where to look (calibration/fourier_refinement.py)
    can instead pass explicit `positions=(x_centers, y_centers)` and a
    boolean `position_mask` of shape (len(y_centers), len(x_centers));
    only positions where the mask is True are acquired, the rest stay
    NaN in the returned map.

    With `use_tqdm` a tqdm progress bar replaces the periodic log lines
    (falling back to them if tqdm isn't installed).

    Returns
    -------
    x_centers, y_centers : 1D arrays of SLM coordinates.
    d2_map : 2D array (len(y_centers), len(x_centers)), NaN where
        localization failed for either sign.
    metadata : dict
    """
    if mode not in MODE_INDICES:
        raise ValueError(
            f"Unknown probe mode {mode!r}. Available: {sorted(MODE_INDICES)}"
        )

    if positions is None:
        x_centers, y_centers = raster_positions(config, step_px, x_range, y_range)
    else:
        x_centers, y_centers = (np.asarray(p, float) for p in positions)
    if position_mask is None:
        position_mask = np.ones((len(y_centers), len(x_centers)), dtype=bool)
    else:
        position_mask = np.asarray(position_mask, dtype=bool)
        if position_mask.shape != (len(y_centers), len(x_centers)):
            raise ValueError(
                f"position_mask shape {position_mask.shape} does not match "
                f"the ({len(y_centers)}, {len(x_centers)}) position grid."
            )
    to_acquire = [(iy, ix) for iy in range(len(y_centers))
                  for ix in range(len(x_centers)) if position_mask[iy, ix]]
    n_pos = len(to_acquire)
    if n_pos == 0:
        raise ValueError("Raster produced no positions — check step_px/ranges.")

    tqdm = None
    if use_tqdm:
        try:
            from tqdm import tqdm
        except ImportError:
            logger.warning("tqdm is not installed; falling back to log-line "
                           "progress (pip install tqdm).")
        else:
            progress_every = 0

    settle = slm_settle_seconds(config)
    logger.info(
        f"Rastering {n_pos} positions on a {len(x_centers)} x {len(y_centers)} "
        f"grid (step {step_px} px, patch {patch_diameter_px} px, mode {mode}, "
        f"amplitude {amplitude_rad / np.pi:.2f}*pi p-v)"
    )
    logger.info(
        f"SLM settle {settle * 1000:.0f} ms/mask; estimated acquisition time "
        f"{format_duration(estimate_duration_s(config, n_pos, exposure_ms))}"
    )

    mmc.setExposure(exposure_ms)
    set_laser_power_mmcore(mmc, config, laser_power_mW)
    laser_on_mmcore(mmc, config)

    d2_map = np.full((len(y_centers), len(x_centers)), np.nan)
    n_failed = 0
    t_start = time.time()
    bar = (tqdm(total=n_pos, desc="Rastering", unit="pos", dynamic_ncols=True)
           if tqdm is not None else None)

    try:
        for k, (iy, ix) in enumerate(to_acquire, start=1):
            xcen, ycen = x_centers[ix], y_centers[iy]
            locs = []
            for sign in (+1, -1):
                mask = zernike_patch_mask(
                    config, (xcen, ycen), patch_diameter_px,
                    mode, amplitude_rad, sign,
                )
                slm.show_mask(mask)
                time.sleep(settle)
                locs.append(
                    localize_psf(mmc.snap(), method=localization_method)
                )

            d2 = squared_separation(locs[0], locs[1])
            d2_map[iy, ix] = d2
            if not np.isfinite(d2):
                n_failed += 1

            if bar is not None:
                bar.set_postfix(failed=n_failed, refresh=False)
                bar.update(1)
            elif progress_every and k % progress_every == 0:
                frac = k / n_pos
                elapsed = time.time() - t_start
                remaining = elapsed / frac - elapsed
                logger.info(
                    f"  {k}/{n_pos} ({100 * frac:.0f}%) — "
                    f"{format_duration(remaining)} remaining, "
                    f"{n_failed} failed so far"
                )
    finally:
        if bar is not None:
            bar.close()
        laser_off_mmcore(mmc, config)
        slm.show_mask(flat_mask(config))

    elapsed = time.time() - t_start
    finished_at = datetime.now().isoformat(sep=" ", timespec="seconds")
    logger.info(
        f"Raster complete in {format_duration(elapsed)}; "
        f"{n_failed}/{n_pos} positions failed localization."
    )
    if n_failed > 0.3 * n_pos:
        logger.warning(
            "More than 30% of positions failed to localize. The map is "
            "probably unreliable — check laser power, exposure, and that "
            "the bead stays inside the ROI for the largest PSF shifts."
        )

    metadata = {
        "probe_mode": mode,
        "amplitude_rad": amplitude_rad,
        "amplitude_over_pi": amplitude_rad / np.pi,
        "patch_diameter_px": patch_diameter_px,
        "step_px": step_px,
        "exposure_ms": exposure_ms,
        "laser_power_mW": laser_power_mW,
        "localization_method": localization_method,
        "slm_settle_s": settle,
        "slm_refresh_rate_hz": config.get("slm", "refresh_rate_hz"),
        "slm_resolution": config.get("slm", "resolution"),
        "camera_roi_xywh": list(mmc.getROI()),
        "pupil_radius_px_assumed": config.get("fourier_plane", "radius_px"),
        "n_positions": n_pos,
        "n_failed_localizations": n_failed,
        "elapsed_s": elapsed,
        "timestamp": finished_at,
        "x_range": x_range,
        "y_range": y_range,
    }
    return x_centers, y_centers, d2_map, metadata


# --- small interactive helpers for the calibration flow ---

def prompt_choice(question, options, default=None):
    """Ask the scientist to pick one of `options` on the terminal."""
    opts = list(options)
    while True:
        shown = "/".join(
            f"[{o}]" if o == default else o for o in opts
        )
        ans = input(f"{question} ({shown}): ").strip()
        if not ans and default is not None:
            return default
        if ans in opts:
            return ans
        print(f"  Please enter one of: {', '.join(opts)}")


def prompt_yes_no(question, default=False):
    d = "y" if default else "n"
    while True:
        ans = input(f"{question} (y/n) [{d}]: ").strip().lower()
        if not ans:
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
