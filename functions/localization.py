"""
functions/localization.py

Point-source (microsphere) localization. Deliberately simple and
robust rather than clever: for alignment we only need a repeatable
centroid, not a physically rigorous PSF fit.

Two methods are available, and the choice matters for this
measurement:

* 'frame_centroid' (default) — background-subtracted intensity
  centroid over the whole ROI. This is the theoretically correct
  estimator here: the intensity centroid of a PSF equals the
  pupil-area-weighted mean of the pupil phase gradient. That identity
  is precisely why a tilt patch shifts the PSF in proportion to how
  much of the pupil it covers, so this method reproduces the expected
  map shape. It assumes the ROI contains one isolated source.

* 'windowed' — brightest pixel, then a centroid in a window around it.
  More robust to stray light or a second bead in the ROI, but it
  tracks the dominant lobe, which is produced by the *unmodulated*
  majority of the pupil and therefore barely moves. Use only as a
  fallback; expect weaker contrast in the map.

Reporting None (rather than a garbage coordinate) when the source is
too dim is important — the map analysis treats those points as NaN
instead of letting them bias the centre estimate.

Never writes to disk.
"""
import numpy as np


def localize_psf(
    frame,
    method="frame_centroid",
    window_half_px=15,
    background_percentile=20.0,
    threshold_fraction=0.25,
    min_snr=3.0,
):
    """
    Locate a single bright point source in a 2D frame.

    Parameters
    ----------
    frame : 2D array
    method : 'frame_centroid' or 'windowed' (see module docstring).
    window_half_px : window half-size, 'windowed' only.
    background_percentile : percentile used as the background level.
    threshold_fraction : pixels below this fraction of the peak are
        excluded from the centroid, which suppresses background bias.
    min_snr : below this peak-to-noise ratio, return None.

    Returns
    -------
    dict with keys 'x', 'y', 'peak', 'background', 'noise', 'snr'
    on success, or None if no source passes the SNR test.
    """
    img = np.asarray(frame, dtype=float)
    if img.ndim != 2:
        raise ValueError(f"Expected a 2D frame, got shape {img.shape}")

    background = float(np.percentile(img, background_percentile))
    noise = float(np.std(img[img <= np.percentile(img, 50.0)]))
    if noise <= 0:
        noise = 1e-9

    sub = img - background
    iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
    peak = float(sub[iy, ix])
    snr = peak / noise
    if snr < min_snr:
        return None

    if method == "frame_centroid":
        yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        w = np.where(sub > threshold_fraction * peak, sub, 0.0)
    elif method == "windowed":
        y0 = max(0, iy - window_half_px)
        y1 = min(img.shape[0], iy + window_half_px + 1)
        x0 = max(0, ix - window_half_px)
        x1 = min(img.shape[1], ix + window_half_px + 1)
        win = sub[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        w = np.where(win > threshold_fraction * peak, win, 0.0)
    else:
        raise ValueError(
            f"Unknown localization method {method!r}; "
            "expected 'frame_centroid' or 'windowed'."
        )

    total = w.sum()
    if total <= 0:
        return None

    cx = float((w * xx).sum() / total)
    cy = float((w * yy).sum() / total)

    return {
        "x": cx,
        "y": cy,
        "peak": peak,
        "background": background,
        "noise": noise,
        "snr": float(snr),
        "method": method,
    }


def squared_separation(loc_a, loc_b):
    """
    Squared distance (in camera px^2) between two localizations.
    Returns NaN if either localization failed.
    """
    if loc_a is None or loc_b is None:
        return float("nan")
    return (loc_a["x"] - loc_b["x"]) ** 2 + (loc_a["y"] - loc_b["y"]) ** 2
