"""
experiments/fourier_plane_refinement.py

Refines the Fourier-plane (pupil) centre of one polarization channel on
the SLM, starting from the estimate an earlier
experiments/fourier_plane_alignment.py run wrote to the config.

Why a second experiment
-----------------------
fourier_plane_alignment rasters the WHOLE SLM, so its acquisition time
grows with 1/step^2 and a fine step is impractical. Once a coarse
centre exists there is no need to look anywhere else: this experiment
rasters only a disk (default radius = the pupil radius) around the
stored centre, at a finer step (default 1/4 of the coarse step), using
the same probe patch and the same d^2 measurement. For example the
default 100 px step over 1920x1152 px is ~200 positions; a 25 px step
over a 360 px radius disk is ~650.

The coarse estimator (support centroid) is not suited to a window this
small - its primary estimate would just be the window centre - so the
centre is instead taken as the point about which the map is best
described as a function of radius alone (calibration/map_analysis.py's
refine_fourier_center).

Flow
----
1. A napari window streams the full camera frame live (flat mask on
   the SLM), with each channel's ROI drawn as a box. Position the
   sample so a well-separated microsphere sits inside the ROI of the
   channel you want, then close the window.
2. Choose the channel (only channels that already have a stored centre
   are offered) and the probe mode in the terminal, and confirm the
   estimated acquisition time.
3. The camera is cropped to that channel's ROI, a verification frame is
   shown in napari to confirm the bead is localized, and you confirm.
4. The windowed raster runs with a progress bar. The raw map is saved,
   a preview (rastered area, fine map, radial profile) is shown, and
   only if you accept does the new centre replace
   fourier_plane.channels.<channel>.center_px in the config (the old
   config is backed up first).

Run with:
    python -m experiments.fourier_plane_refinement
"""
import numpy as np

from core.config import load_config
from core.file_io import (
    save_alignment_map,
    save_figure,
    save_frame_tiff,
    write_fourier_center,
)
from core.logging_setup import setup_logger
from core.mmcore import load_mmcore
from core.session import Session
from functions.mmcore_camera import set_channel_roi
from functions.mmcore_slm import MMCoreSLM
from functions.napari_preview import show_verification_frame
from calibration.fourier_alignment import (
    acquire_d2_map,
    estimate_duration_s,
    format_duration,
    preview_sample_positioning,
    prompt_choice,
    prompt_yes_no,
    verify_point_source,
)
from calibration.fourier_refinement import calibrated_channels, stored_center
from calibration.map_analysis import refine_fourier_center
from calibration.map_preview import build_refinement_figure
from functions.phase_masks import local_raster_positions

# ---- experiment parameters (override the config defaults here) ----
REFINE_FACTOR = 4          # fine step = coarse step_px / REFINE_FACTOR
STEP_PX = None             # None -> fourier_alignment.step_px / REFINE_FACTOR
WINDOW_RADIUS_PX = None    # None -> fourier_plane.radius_px (the pupil radius)
PATCH_DIAMETER_PX = None   # None -> use config (keep equal to the coarse run)
AMPLITUDE_RAD = None       # None -> use the per-mode config default
EXPOSURE_MS = None         # None -> use config
LASER_POWER_MW = None      # None -> use config

PROBE_MODES = ["tilt_x", "tilt_y", "defocus"]


def main():
    config = load_config()
    fa = config["fourier_alignment"]

    step_px = STEP_PX or fa["step_px"] / REFINE_FACTOR
    window_r = WINDOW_RADIUS_PX or config.get("fourier_plane", "radius_px")
    patch_d = PATCH_DIAMETER_PX or fa["patch_diameter_px"]
    exposure_ms = EXPOSURE_MS or fa["exposure_ms"]
    laser_power = LASER_POWER_MW or fa["laser_power_mW"]
    loc_method = fa.get("localization_method", "frame_centroid")

    if not window_r:
        print("No raster radius: set WINDOW_RADIUS_PX or "
              "fourier_plane.radius_px in the config.")
        return
    if not calibrated_channels(config):
        print("No channel has a stored Fourier-plane centre yet. Run "
              "experiments/fourier_plane_alignment.py first.")
        return

    session = Session(config, script_name="fourier_plane_refinement")
    logger = setup_logger(session.path)
    logger.info(f"Run folder: {session.path}")

    mmc = load_mmcore(config)
    logger.info("Connected to Micro-Manager via pymmcore-plus.")

    with MMCoreSLM(mmc, config) as slm:
        try:
            acquired = _acquire(mmc, config, slm, logger, session, step_px,
                                window_r, patch_d, exposure_ms, laser_power,
                                loc_method)
        finally:
            mmc.clearROI()
    if acquired is not None:
        _analyse(config, logger, session, window_r, *acquired)


def _acquire(mmc, config, slm, logger, session, step_px, window_r, patch_d,
             exposure_ms, laser_power, loc_method):
    """Steps 1-4a: preview, choices, verification, windowed raster."""
    # --- Step 1: live preview to position the sample ---
    preview_sample_positioning(mmc, config, slm, logger, exposure_ms,
                               laser_power)

    # --- Step 2: scientist-supplied choices ---
    channels = calibrated_channels(config)
    channel = prompt_choice("Which polarization channel should be refined?",
                            channels, default=channels[0] if len(channels) == 1
                            else None)
    previous = stored_center(config, channel)

    stored_mode = config.get("fourier_plane", "channels", channel,
                             "calibration_mode")
    mode = prompt_choice("Which Zernike probe mode?", PROBE_MODES,
                         default=stored_mode if stored_mode in PROBE_MODES
                         else "tilt_x")
    amplitude = AMPLITUDE_RAD or config["fourier_alignment"]["amplitude_rad_by_mode"][mode]
    logger.info(f"Channel '{channel}', probe mode '{mode}', "
                f"amplitude {amplitude / np.pi:.2f}*pi peak-to-valley")
    logger.info(f"Previous centre: ({previous[0]:.1f}, {previous[1]:.1f}) SLM px")

    xs, ys, window_mask = local_raster_positions(config, previous, window_r,
                                                 step_px)
    n_pos = int(window_mask.sum())
    est = estimate_duration_s(config, n_pos, exposure_ms)
    print(f"\nRefining around ({previous[0]:.1f}, {previous[1]:.1f}) SLM px: "
          f"radius {window_r:g} px, step {step_px:g} px")
    print(f"{n_pos} raster positions -> ~{format_duration(est)} "
          f"of acquisition.")
    full_disk = np.pi * (window_r / step_px) ** 2
    if n_pos < 0.9 * full_disk:
        print("  NOTE: the window runs off the edge of the SLM, so part of "
              "it is not rastered.")
    if not prompt_yes_no("Proceed?", default=True):
        logger.info("Aborted before acquisition at the scientist's request.")
        return None

    roi = set_channel_roi(mmc, config, channel)
    logger.info(f"Camera ROI set to channel '{channel}': "
                f"x={roi[0]}, y={roi[1]}, size={roi[2]} px")

    # --- Step 3: verify the point source inside the ROI ---
    frame, loc = verify_point_source(
        mmc, config, slm, logger, exposure_ms, laser_power, loc_method
    )
    save_frame_tiff(session.path, frame,
                    {"purpose": "point source verification",
                     "channel": channel,
                     "camera_roi_xywh": list(roi),
                     "exposure_ms": exposure_ms,
                     "laser_power_mW": laser_power,
                     "localization": loc},
                    filename="verification_frame.tif")

    show_verification_frame(
        frame, loc,
        title=f"ReflEx3D — verification, channel '{channel}' (close to continue)",
    )
    if loc is None:
        print("No point source was localized in the verification frame.")
    if not prompt_yes_no("Is the point source visible and correctly "
                         "localized?", default=False):
        logger.info("Aborted at point-source verification.")
        print("Aborted. Adjust the sample/ROI/exposure and run again.")
        return None

    # --- Step 4a: windowed fine raster ---
    x_centers, y_centers, d2_map, acq_meta = acquire_d2_map(
        mmc, config, slm, logger,
        mode=mode,
        amplitude_rad=amplitude,
        patch_diameter_px=patch_d,
        step_px=step_px,
        exposure_ms=exposure_ms,
        laser_power_mW=laser_power,
        localization_method=loc_method,
        positions=(xs, ys),
        position_mask=window_mask,
        use_tqdm=True,
    )

    # Save the raw map before asking for approval, as in the coarse
    # experiment: approval gates the config write-back, not the data.
    acq_meta.update({
        "channel": channel,
        "refinement": True,
        "previous_center_px": list(previous),
        "window_radius_px": window_r,
    })
    map_path = save_alignment_map(session.path, x_centers, y_centers,
                                  d2_map, acq_meta)
    logger.info(f"Raw map saved to {map_path}")
    return channel, mode, previous, x_centers, y_centers, d2_map, acq_meta


def _analyse(config, logger, session, window_r, channel, mode, previous,
             x_centers, y_centers, d2_map, acq_meta):
    """Steps 4b-6: estimate the centre, preview, approve + write back."""
    result = refine_fourier_center(d2_map, x_centers, y_centers, mode,
                                   previous, window_r)

    cx, cy = result["center_x"], result["center_y"]
    for w in result["warnings"]:
        logger.warning(w)

    if not np.isfinite(cx) or not np.isfinite(cy):
        print("\nCentre estimation failed. The raw map has been saved for "
              "re-analysis; see the warnings in the log.")
        return
    logger.info(f"Refined centre: ({cx:.1f}, {cy:.1f}) SLM px, "
                f"{result['shift_px']:.1f} px from the previous one")

    fig = build_refinement_figure(x_centers, y_centers, d2_map, result,
                                  config, channel)
    preview_path = save_figure(session.path, fig,
                               filename="fourier_refinement_preview.png")
    logger.info(f"Preview saved to {preview_path}")
    _show_figure(fig)

    print(f"\nChannel '{channel}' Fourier-plane centre:")
    print(f"  previous: ({previous[0]:.1f}, {previous[1]:.1f}) SLM px")
    print(f"  refined:  ({cx:.1f}, {cy:.1f}) SLM px  "
          f"(shift {result['shift_px']:.1f} px, radial model explains "
          f"{100 * result['explained_fraction']:.0f}% of the map)")
    for w in result["warnings"]:
        print(f"  WARNING: {w}")

    if not prompt_yes_no("Accept the refined centre and write it to the "
                         "config?", default=False):
        logger.info("Scientist rejected the estimate; config not modified.")
        print("Config left unchanged. The map is saved and can be "
              "re-analysed without re-acquiring.")
        return

    cfg_path, backup_path = write_fourier_center(
        config.path, channel, (cx, cy),
        extra={
            "calibrated_on": acq_meta.get("timestamp") or None,
            "calibration_mode": mode,
            "calibration_run": str(session.path),
        },
    )
    logger.info(f"Wrote centre to {cfg_path} (backup at {backup_path})")
    print(f"Written to {cfg_path}\nPrevious config backed up at {backup_path}")


def _show_figure(fig):
    """Show a figure, degrading gracefully on a headless machine."""
    try:
        import matplotlib.pyplot as plt
        plt.show()
    except Exception as exc:  # no display available
        print(f"(Could not open a plot window: {exc}. "
              f"The saved PNG in the run folder has the same content.)")


if __name__ == "__main__":
    main()
