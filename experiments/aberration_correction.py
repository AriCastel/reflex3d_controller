"""
experiments/aberration_correction.py

Sensorless adaptive optics on a guide star (an isolated microsphere):
the SLM phase over one channel's pupil is a sum of Zernike modes, and
their coefficients are optimized by SPSA stochastic gradient descent
to minimize a PSF metric (functions/psf_metrics.py, chosen in the
config). See functions/adaptive_optics.py for the algorithm.

Flow
----
1. Live full-frame preview (flat mask) with the channel ROIs boxed -
   position the guide star, then close the window.
2. Choose the channel in the terminal. Its Fourier-plane centre must
   already be calibrated (experiments/fourier_plane_alignment.py).
3. The camera is cropped to the channel ROI, the guide star is
   localized and shown for confirmation, then the camera is cropped to
   a small square around it and a dark frame is taken (laser off).
4. SPSA runs. Then the flat and corrected masks are snapped back to
   back, and the run's figure (metric vs iteration, original vs
   corrected PSF, final phase, coefficients) is saved and shown, and
   the coefficients printed.

Run with:
    python -m experiments.aberration_correction
"""
from datetime import datetime

from core.config import load_config
from core.file_io import save_ao_result, save_figure
from core.logging_setup import setup_logger
from core.mmcore import load_mmcore
from core.session import Session
from calibration.fourier_alignment import (
    format_duration,
    preview_sample_positioning,
    prompt_choice,
    prompt_yes_no,
    slm_settle_seconds,
    verify_point_source,
)
from functions.adaptive_optics import (
    ZernikePupil,
    acquire_dark_frame,
    build_ao_figure,
    coefficient_table,
    run_aberration_correction,
    set_guide_star_roi,
)
from functions.mmcore_camera import set_channel_roi
from functions.mmcore_laser import laser_off_mmcore, laser_on_mmcore, set_laser_power_mmcore
from functions.mmcore_slm import MMCoreSLM
from functions.napari_preview import show_verification_frame
from functions.psf_metrics import make_metric

# ---- experiment parameters (None -> use adaptive_optics.* from config) ----
ITERATIONS = None
EXPOSURE_MS = None
LASER_POWER_MW = None
METRIC = None             # e.g. {"name": "encircled_energy", "params": {"radius_px": 4}}


def main():
    config = load_config()
    ao = config["adaptive_optics"]
    iterations = ITERATIONS or ao["iterations"]
    exposure_ms = EXPOSURE_MS or ao["exposure_ms"]
    laser_power = LASER_POWER_MW or ao["laser_power_mW"]
    metric_cfg = METRIC or ao["metric"]
    noll = ao["noll_indices"]
    settle = slm_settle_seconds(config)

    session = Session(config, script_name="aberration_correction")
    logger = setup_logger(session.path)
    logger.info(f"Run folder: {session.path}")

    mmc = load_mmcore(config)
    logger.info("Connected to Micro-Manager via pymmcore-plus.")

    with MMCoreSLM(mmc, config) as slm:
        try:
            out = _acquire(mmc, config, ao, slm, logger, iterations,
                           exposure_ms, laser_power, metric_cfg, noll, settle)
        finally:
            laser_off_mmcore(mmc, config)
            mmc.clearROI()
    if out is None:
        return
    channel, result, gs_roi = out

    metadata = {
        "channel": channel,
        "finished_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "metric": metric_cfg,
        "noll_indices": noll,
        "coefficients_rad_rms": {str(j): float(c) for j, c in zip(noll, result["coefficients"])},
        "metric_initial": result["metric_initial"],
        "metric_original_end": result["metric_original"],
        "metric_corrected_end": result["metric_corrected"],
        "power_ratio_corrected_over_original": result["power_ratio"],
        "iterations_run": len(result["history"]["iteration"]),
        "n_snaps": result["n_snaps"],
        "elapsed_s": result["elapsed_s"],
        "exposure_ms": exposure_ms,
        "laser_power_mW": laser_power,
        "guide_star_roi_xywh": list(gs_roi),
        "slm_settle_s": settle,
        "optimizer": {k: ao[k] for k in ("gain", "perturbation_rad", "momentum",
                                          "max_abs_coefficient_rad", "patience", "seed")},
        "fourier_center_px": config.get("fourier_plane", "channels", channel, "center_px"),
        "pupil_radius_px": config.get("fourier_plane", "radius_px"),
    }
    path = save_ao_result(session.path, result, metadata)
    logger.info(f"Results saved to {path}")

    fig = build_ao_figure(result, noll, metric_cfg["name"], channel)
    logger.info(f"Figure saved to {save_figure(session.path, fig, 'ao_correction.png')}")

    print(f"\nFinal Zernike coefficients, channel '{channel}':")
    print(coefficient_table(noll, result["coefficients"]))
    print(f"\nMetric {metric_cfg['name']}: {result['metric_original']:.4g} (flat) -> "
          f"{result['metric_corrected']:.4g} (corrected); "
          f"corrected/original power {result['power_ratio']:.2f}")
    _show_figure()


def _acquire(mmc, config, ao, slm, logger, iterations, exposure_ms,
             laser_power, metric_cfg, noll, settle):
    # --- Step 1: position the guide star ---
    preview_sample_positioning(mmc, config, slm, logger, exposure_ms, laser_power)

    # --- Step 2: channel ---
    channels = list(config.get("fourier_plane", "channels", default={}).keys())
    channel = prompt_choice("Which polarization channel should be corrected?",
                            channels or ["left", "right"])
    pupil = ZernikePupil(config, channel, noll)
    metric = make_metric(metric_cfg["name"], metric_cfg.get("params"))
    logger.info(f"Channel '{channel}', {len(noll)} modes (Noll {noll}), "
                f"metric {metric_cfg}")

    per_iter = 2 * (settle + exposure_ms / 1000.0 + 0.025)
    print(f"\n{iterations} iterations x 2 snaps -> ~{format_duration(iterations * per_iter)} "
          f"of illumination on the guide star.")
    if not prompt_yes_no("Proceed?", default=True):
        logger.info("Aborted before acquisition at the scientist's request.")
        return None

    # --- Step 3: find the guide star and crop to it ---
    roi = set_channel_roi(mmc, config, channel)
    frame, loc = verify_point_source(mmc, config, slm, logger, exposure_ms,
                                     laser_power, "windowed")
    show_verification_frame(
        frame, loc, title=f"ReflEx3D — guide star, channel '{channel}' (close to continue)")
    if loc is None or not prompt_yes_no("Is this the guide star?", default=False):
        logger.info("Aborted at guide-star verification.")
        return None

    gs_roi = set_guide_star_roi(mmc, (roi[0] + loc["x"], roi[1] + loc["y"]),
                                ao["guide_star_roi_px"])
    logger.info(f"Camera cropped to the guide star: ROI {gs_roi}")
    mmc.setExposure(exposure_ms)
    dark = acquire_dark_frame(mmc, ao["dark_frames"])   # laser is off here

    # --- Step 4: optimize ---
    set_laser_power_mmcore(mmc, config, laser_power)
    laser_on_mmcore(mmc, config)
    result = run_aberration_correction(
        mmc, config, slm, pupil, metric, dark, logger,
        settle_s=settle,
        iterations=iterations,
        gain=ao["gain"],
        perturbation=ao["perturbation_rad"],
        momentum=ao["momentum"],
        max_abs=ao["max_abs_coefficient_rad"],
        patience=ao["patience"],
        seed=ao["seed"],
        laser_on=lambda: laser_on_mmcore(mmc, config),
        laser_off=lambda: laser_off_mmcore(mmc, config),
        blank_laser_during_settle=ao["blank_laser_during_settle"],
    )
    return channel, result, gs_roi


def _show_figure():
    try:
        import matplotlib.pyplot as plt
        plt.show()
    except Exception as exc:  # no display available
        print(f"(Could not open a plot window: {exc}. "
              f"The saved PNG in the run folder has the same content.)")


if __name__ == "__main__":
    main()
