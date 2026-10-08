"""
functions/mmcore_camera.py

pymmcore-plus camera helpers. Exposure and snapping need no wrapper
(CMMCorePlus exposes `setExposure`/`snap` directly); what lives here
is the per-channel camera ROI. The legacy workflow drew that ROI by
hand in the Micro-Manager GUI; here it's read from
`camera.channel_rois` in the config instead, since each polarization
channel always lands on the same part of the sensor. Never writes to
disk.
"""


def get_channel_roi(config, channel):
    """
    The square camera ROI for one polarization channel, as
    (x, y, width, height) in full-frame camera pixels, where (x, y) is
    the top-left corner - the same convention as CMMCorePlus.setROI.
    """
    roi = config.get("camera", "channel_rois", channel)
    if roi is None:
        raise KeyError(
            f"No camera.channel_rois.{channel} entry in the config - add "
            f"one (position_px + size_px) for this channel."
        )
    x, y = roi["position_px"]
    size = roi["size_px"]
    if x is None or y is None or size is None:
        raise ValueError(
            f"camera.channel_rois.{channel} is incomplete: position_px="
            f"{roi['position_px']}, size_px={size}."
        )
    return int(x), int(y), int(size), int(size)


def get_all_channel_rois(config):
    """{channel: (x, y, width, height)} for every ROI in the config."""
    rois = config.get("camera", "channel_rois", default={})
    return {
        channel: get_channel_roi(config, channel)
        for channel in rois
        if not channel.startswith("_")
    }


def set_channel_roi(mmc, config, channel):
    """
    Crop the camera to `channel`'s ROI, after checking that it fits on
    the sensor. Returns the (x, y, width, height) that was applied.
    """
    x, y, width, height = get_channel_roi(config, channel)
    mmc.clearROI()
    sensor_w, sensor_h = mmc.getImageWidth(), mmc.getImageHeight()
    if x < 0 or y < 0 or x + width > sensor_w or y + height > sensor_h:
        raise ValueError(
            f"camera.channel_rois.{channel} = (x={x}, y={y}, size={width}) "
            f"doesn't fit on the {sensor_w}x{sensor_h} px sensor."
        )
    mmc.setROI(x, y, width, height)
    return x, y, width, height
