"""
functions/phase_masks.py

Builds full-SLM uint8 phase masks. For the Fourier-plane alignment
routine, the mask is a small circular *patch* carrying a Zernike ramp,
placed on an otherwise flat background — only the part of the patch
that overlaps the pupil affects the PSF, which is exactly what makes
the raster informative.

Never writes to disk.
"""
import numpy as np

from functions.zernike import mode_surface


def phase_to_grey(phase_rad, grey_level_2pi):
    """
    Wrap a phase map into [0, 2*pi) and map it onto the SLM's 8-bit
    grey levels using the device's calibrated 2*pi level.
    """
    wrapped = np.mod(phase_rad, 2.0 * np.pi)
    grey = np.round(wrapped / (2.0 * np.pi) * grey_level_2pi)
    return np.mod(grey, 256).astype(np.uint8)


def zernike_patch_mask(
    config,
    center_xy,
    patch_diameter_px,
    mode,
    amplitude_rad=2.0 * np.pi,
    sign=+1,
):
    """
    Full-SLM mask: flat background with one circular Zernike patch.

    Parameters
    ----------
    config : MicroscopeConfig
    center_xy : (x, y) patch centre in SLM pixel coordinates.
    patch_diameter_px : diameter of the circular patch, in SLM pixels.
    mode : 'tilt_x', 'tilt_y' or 'defocus'.
    amplitude_rad : requested peak-to-valley phase across the patch.
        The mode surface is rescaled so that (max - min) over the patch
        equals this value, which makes `2*pi` mean a literal 0->2*pi
        ramp regardless of which mode is used.
    sign : +1 or -1. Flipping the sign reverses the ramp (0->2*pi
        becomes 2*pi->0), which is what reverses the PSF shift
        direction.

    Returns
    -------
    uint8 array of shape (slm_height, slm_width).
    """
    if sign not in (+1, -1):
        raise ValueError(f"sign must be +1 or -1, got {sign}")

    width, height = config.get("slm", "resolution", default=[1920, 1080])
    flat_value = config.get("slm", "flat_value", default=128)
    grey_2pi = config.get("slm", "grey_level_2pi", default=255)

    mask = np.full((height, width), flat_value, dtype=np.uint8)

    cx, cy = float(center_xy[0]), float(center_xy[1])
    r_patch = patch_diameter_px / 2.0

    # Bounding box of the patch, clipped to the SLM (patches near the
    # edge are simply truncated).
    x0 = int(np.floor(cx - r_patch))
    x1 = int(np.ceil(cx + r_patch)) + 1
    y0 = int(np.floor(cy - r_patch))
    y1 = int(np.ceil(cy + r_patch)) + 1
    xs0, xs1 = max(0, x0), min(width, x1)
    ys0, ys1 = max(0, y0), min(height, y1)
    if xs0 >= xs1 or ys0 >= ys1:
        return mask  # patch entirely off-screen

    yy, xx = np.mgrid[ys0:ys1, xs0:xs1]
    dx = (xx - cx) / r_patch
    dy = (yy - cy) / r_patch
    rho = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dy, dx)

    inside = rho <= 1.0
    if not np.any(inside):
        return mask

    surface = mode_surface(mode, rho, theta)

    # Rescale so peak-to-valley across the patch == amplitude_rad.
    s_in = surface[inside]
    span = float(s_in.max() - s_in.min())
    if span <= 0:
        return mask
    phase = sign * amplitude_rad * (surface - s_in.min()) / span

    patch_grey = phase_to_grey(phase, grey_2pi)
    region = mask[ys0:ys1, xs0:xs1]
    region[inside] = patch_grey[inside]
    mask[ys0:ys1, xs0:xs1] = region
    return mask


def raster_positions(config, step_px, x_range=None, y_range=None):
    """
    Evenly spaced patch-centre positions covering the SLM.

    `step_px` is the sampling interval in SLM pixels — smaller means a
    denser map and a proportionally longer acquisition.

    x_range / y_range optionally restrict the raster to a sub-area,
    which is useful for a fast coarse pass followed by a fine pass
    around the coarse estimate.

    Returns
    -------
    x_centers, y_centers : 1D arrays of patch centre coordinates.
    """
    width, height = config.get("slm", "resolution", default=[1920, 1080])
    x_lo, x_hi = x_range if x_range is not None else (0, width)
    y_lo, y_hi = y_range if y_range is not None else (0, height)

    x_centers = np.arange(x_lo + step_px / 2.0, x_hi, step_px)
    y_centers = np.arange(y_lo + step_px / 2.0, y_hi, step_px)
    return x_centers, y_centers
