"""
experiments/fourier_plane_alignment.py

Locates the centre of the Fourier plane (pupil) on the SLM for one
polarization channel, so that phase masks can be correctly registered
to it in later AO / EDOF work.

How it works
------------
A small circular Zernike probe patch is rastered across the SLM. At
each position it is shown twice, once with +amplitude (0 -> 2*pi) and
once with -amplitude (2*pi -> 0), and the point source is localized in
each frame. The squared separation d^2 between the two localizations
builds a map over the SLM:

  * tilt    -> d^2 is MAXIMUM at the centre (patch covers most pupil)
  * defocus -> d^2 is MINIMUM at the centre (no net tilt when centred)

Either way d^2 -> 0 where the patch misses the pupil entirely.

Flow
----
1. A napari window streams the full camera frame live (flat mask on
   the SLM), with each channel's ROI from `camera.channel_rois` drawn
   as a box. Position the sample so a well-separated microsphere sits
   inside the ROI of the channel you want, then close the window.
2. Choose the channel and the probe mode in the terminal.
3. The camera is cropped to that channel's ROI, a verification frame
   is shown in napari to confirm the bead is localized, and the raster
   runs.

Runs on the pymmcore-plus stack: point `pymmcore_plus.device_adapter_path`
/ `pymmcore_plus.system_config_path` at your Micro-Manager install and
hardware config (including the Generic SLM device, `slm.device_label`).

Run with:
    python -m experiments.fourier_plane_alignment
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
from calibration.map_analysis import consistency_check, estimate_fourier_center
from calibration.map_preview import build_map_figure
from functions.phase_masks import raster_positions

# ---- experiment parameters (override the config defaults here) ----
STEP_PX = None            # None -> use fourier_alignment.step_px from config
PATCH_DIAMETER_PX = None  # None -> use config
AMPLITUDE_RAD = None      # None -> use the per-mode config default
EXPOSURE_MS = None        # None -> use config
LASER_POWER_MW = None     # None -> use config

# Optional: restrict the raster to a sub-area for a fast second pass,
# e.g. X_RANGE = (700, 1300). None rasters the full SLM.
X_RANGE = None
Y_RANGE = None

PROBE_MODES = ["tilt_x", "tilt_y", "defocus"]


def main():
    config = load_config()
    fa = config["fourier_alignment"]

    step_px = STEP_PX or fa["step_px"]
    patch_d = PATCH_DIAMETER_PX or fa["patch_diameter_px"]
    exposure_ms = EXPOSURE_MS or fa["exposure_ms"]
    laser_power = LASER_POWER_MW or fa["laser_power_mW"]
    loc_method = fa.get("localization_method", "frame_centroid")

    session = Session(config, script_name="fourier_plane_alignment")
    logger = setup_logger(session.path)
    logger.info(f"Run folder: {session.path}")

    mmc = load_mmcore(config)
    logger.info("Connected to Micro-Manager via pymmcore-plus.")

    with MMCoreSLM(mmc, config) as slm:
        try:
            acquired = _acquire(mmc, config, slm, logger, session,
                                step_px, patch_d, exposure_ms,
                                laser_power, loc_method)
        finally:
            mmc.clearROI()
    if acquired is not None:
        _analyse(config, fa, logger, session, patch_d, *acquired)


def _acquire(mmc, config, slm, logger, session, step_px, patch_d,
             exposure_ms, laser_power, loc_method):
    """Steps 1-3: preview, choices, verification, raster."""
    # --- Step 1: live preview to position the sample ---
    preview_sample_positioning(mmc, config, slm, logger, exposure_ms,
                               laser_power)

    # --- Step 2: scientist-supplied choices ---
    channels = list(config.get("fourier_plane", "channels", default={}).keys())
    channel = prompt_choice("Which polarization channel should be aligned?",
                            channels or ["left", "right"])
    mode = prompt_choice("Which Zernike probe mode?", PROBE_MODES,
                         default="tilt_x")
    amplitude = AMPLITUDE_RAD or config["fourier_alignment"]["amplitude_rad_by_mode"][mode]
    logger.info(f"Channel '{channel}', probe mode '{mode}', "
                f"amplitude {amplitude / np.pi:.2f}*pi peak-to-valley")

    xs_pre, ys_pre = raster_positions(config, step_px, X_RANGE, Y_RANGE)
    n_pos = len(xs_pre) * len(ys_pre)
    est = estimate_duration_s(config, n_pos, exposure_ms)
    print(f"\n{n_pos} raster positions -> ~{format_duration(est)} "
          f"of acquisition.")
    if not prompt_yes_no("Proceed?", default=True):
        logger.info("Aborted before acquisition at the scientist's request.")
        return None

    roi = set_channel_roi(mmc, config, channel)
    logger.info(f"Camera ROI set to channel '{channel}': "
                f"x={roi[0]}, y={roi[1]}, size={roi[2]} px")

    # --- Step 3a: verify the point source inside the ROI ---
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

    # --- Step 3b: raster the SLM ---
    x_centers, y_centers, d2_map, acq_meta = acquire_d2_map(
        mmc, config, slm, logger,
        mode=mode,
        amplitude_rad=amplitude,
        patch_diameter_px=patch_d,
        step_px=step_px,
        exposure_ms=exposure_ms,
        laser_power_mW=laser_power,
        localization_method=loc_method,
        x_range=X_RANGE,
        y_range=Y_RANGE,
    )

    # Save the raw map straight away, BEFORE asking for approval. A
    # raster can take a long time and losing it to a rejected fit (or a
    # crash in plotting) would be painful; approval gates the config
    # write-back, not the raw data.
    acq_meta["channel"] = channel
    map_path = save_alignment_map(session.path, x_centers, y_centers,
                                  d2_map, acq_meta)
    logger.info(f"Raw map saved to {map_path}")
    return channel, mode, x_centers, y_centers, d2_map, acq_meta


def _analyse(config, fa, logger, session, patch_d, channel, mode,
             x_centers, y_centers, d2_map, acq_meta):
    """Steps 4-6: estimate the centre, preview, approve + write back."""
    result = estimate_fourier_center(
        d2_map, x_centers, y_centers, mode,
        support_fraction=fa.get("support_fraction", 0.15),
    )
    result["patch_diameter_px"] = patch_d
    result = consistency_check(result, config)

    cx, cy = result["center_x"], result["center_y"]
    logger.info(f"Estimated centre: ({cx:.1f}, {cy:.1f}) SLM px "
                f"via {result['method']}")
    for w in result["warnings"]:
        logger.warning(w)

    if not np.isfinite(cx) or not np.isfinite(cy):
        print("\nCentre estimation failed. The raw map has been saved for "
              "re-analysis; see the warnings in the log.")
        return

    fig = build_map_figure(x_centers, y_centers, d2_map, result,
                           config, channel)
    preview_path = save_figure(session.path, fig)
    logger.info(f"Preview saved to {preview_path}")
    _show_figure(fig)

    print(f"\nEstimated Fourier-plane centre for channel '{channel}': "
          f"({cx:.1f}, {cy:.1f}) SLM px  [{result['method']}]")
    print(f"Expected a {result['expected_extremum']} at the centre "
          f"for the '{mode}' probe.")
    for w in result["warnings"]:
        print(f"  WARNING: {w}")

    if not prompt_yes_no("Accept this centre and write it to the config?",
                         default=False):
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
