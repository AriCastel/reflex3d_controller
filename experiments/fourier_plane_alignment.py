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

Before running
--------------
Select a ROI in Micro-Manager showing ONLY ONE polarization channel
(one channel occupies the left half of the sensor, the other the
right), with a well-separated microsphere inside it. You will be asked
which channel it is, and the result is stored per channel.

Run with:
    python -m experiments.fourier_plane_alignment
"""
import numpy as np
from pycromanager import Core

from core.config import load_config
from core.file_io import (
    save_alignment_map,
    save_figure,
    save_frame_tiff,
    write_fourier_center,
)
from core.logging_setup import setup_logger
from core.session import Session
from functions.slm import SLMDisplay
from calibration.fourier_alignment import (
    acquire_d2_map,
    estimate_duration_s,
    format_duration,
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

    # --- scientist-supplied choices, before any acquisition ---
    channels = list(config.get("fourier_plane", "channels", default={}).keys())
    channel = prompt_choice("Which polarization channel is in the ROI?",
                            channels or ["left", "right"])
    mode = prompt_choice("Which Zernike probe mode?", PROBE_MODES,
                         default="tilt_x")
    amplitude = AMPLITUDE_RAD or fa["amplitude_rad_by_mode"][mode]

    session = Session(config, script_name="fourier_plane_alignment")
    logger = setup_logger(session.path)
    logger.info(f"Run folder: {session.path}")
    logger.info(f"Channel '{channel}', probe mode '{mode}', "
                f"amplitude {amplitude / np.pi:.2f}*pi peak-to-valley")

    xs_pre, ys_pre = raster_positions(config, step_px, X_RANGE, Y_RANGE)
    n_pos = len(xs_pre) * len(ys_pre)
    est = estimate_duration_s(config, n_pos, exposure_ms)
    print(f"\n{n_pos} raster positions -> ~{format_duration(est)} "
          f"of acquisition.")
    if not prompt_yes_no("Proceed?", default=True):
        logger.info("Aborted before acquisition at the scientist's request.")
        return

    core = Core()
    logger.info("Connected to Micro-Manager.")

    # --- Step 1: verify the point source (SLM window opened and closed
    # around this step so only one Tk root exists at a time, which keeps
    # the matplotlib preview windows from clashing with the SLM window) ---
    with SLMDisplay(config) as slm:
        frame, loc = verify_point_source(
            core, config, slm, logger, exposure_ms, laser_power, loc_method
        )

    save_frame_tiff(session.path, frame,
                    {"purpose": "point source verification",
                     "exposure_ms": exposure_ms,
                     "laser_power_mW": laser_power,
                     "localization": loc},
                    filename="verification_frame.tif")

    _show_verification(frame, loc)
    if loc is None:
        print("No point source was localized in the verification frame.")
    if not prompt_yes_no("Is the point source visible and correctly "
                         "localized?", default=False):
        logger.info("Aborted at point-source verification.")
        print("Aborted. Adjust the sample/ROI/exposure and run again.")
        return

    # --- Step 2: raster the SLM ---
    with SLMDisplay(config) as slm:
        x_centers, y_centers, d2_map, acq_meta = acquire_d2_map(
            core, config, slm, logger,
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

    # --- Step 3: estimate the centre ---
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

    # --- Step 4: preview ---
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

    # --- Step 5: approve and write back to the config ---
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


def _show_verification(frame, loc):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 6))
    vmin, vmax = np.percentile(frame, [1, 99.9])
    ax.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax)
    if loc is not None:
        ax.plot(loc["x"], loc["y"], "r+", ms=20, mew=2)
        ax.set_title(f"Verification — source at ({loc['x']:.1f}, "
                     f"{loc['y']:.1f}), SNR {loc['snr']:.1f}")
    else:
        ax.set_title("Verification — NO source localized")
    ax.set_xlabel("camera x (px)")
    ax.set_ylabel("camera y (px)")
    _show_figure(fig)


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
