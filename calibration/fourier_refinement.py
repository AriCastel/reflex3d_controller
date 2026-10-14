"""
calibration/fourier_refinement.py

Small helpers for the Fourier-plane *refinement* pass
(experiments/fourier_plane_refinement.py): reading back the centre an
earlier fourier_plane_alignment run wrote to the config, and sizing the
fine raster window around it.

The acquisition itself is calibration/fourier_alignment.py's
acquire_d2_map (given an explicit window of positions), and the centre
estimate is calibration/map_analysis.py's refine_fourier_center.
"""


def stored_center(config, channel):
    """
    The (x, y) centre in SLM px stored for `channel`, or None if that
    channel has not been calibrated yet (missing or null entries).
    """
    center = config.get("fourier_plane", "channels", channel, "center_px")
    if not center or len(center) != 2 or any(c is None for c in center):
        return None
    return float(center[0]), float(center[1])


def calibrated_channels(config):
    """Names of the channels that already have a stored centre."""
    channels = config.get("fourier_plane", "channels", default={}) or {}
    return [name for name in channels
            if isinstance(channels[name], dict) and stored_center(config, name)]
