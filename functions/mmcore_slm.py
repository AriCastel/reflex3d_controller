"""
functions/mmcore_slm.py

pymmcore-plus equivalent of functions/slm.py: instead of a plain
secondary-display tkinter window, the SLM is addressed as a
Micro-Manager device (the "Generic SLM: Spatial light modulator
controlled through computer graphics" device) through a CMMCorePlus
instance's native (camelCase) API. Never writes to disk.

Micro-Manager (via pymmcore-plus) owns the fullscreen window on the
SLM's monitor; this module only pushes uint8 pixel buffers to it, the
same way functions/mmcore_laser.py addresses the laser through `mmc`
instead of pycromanager's `core`.
"""
import numpy as np


class MMCoreSLM:
    """
    Thin wrapper around Micro-Manager's Generic SLM device, addressed
    through a CMMCorePlus instance.

    Usage:
        with MMCoreSLM(mmc, config) as slm:
            slm.show_mask(flat_mask(config))
            ... run the experiment while the mask stays up ...
    """

    def __init__(self, mmc, config):
        self.mmc = mmc
        self.config = config
        self.label = config.get("slm", "device_label", default="SLM")

    def __enter__(self):
        core_width = self.mmc.getSLMWidth(self.label)
        core_height = self.mmc.getSLMHeight(self.label)
        cfg_width, cfg_height = self.config.get(
            "slm", "resolution", default=[core_width, core_height]
        )
        if (core_width, core_height) != (cfg_width, cfg_height):
            raise RuntimeError(
                f"slm.resolution in the config is {[cfg_width, cfg_height]}, "
                f"but Micro-Manager reports the '{self.label}' SLM device as "
                f"{core_width}x{core_height}. Fix config/microscope_config.json "
                f"(or the Generic SLM device's resolution in the MM hardware "
                f"config) so the two agree."
            )
        return self

    def show_mask(self, mask_array):
        """Push a new uint8 (H, W) mask array to the SLM and display it."""
        # Kept 2D: the C++ core only checks the byte count, but Python
        # (UniMMCore) SLM devices require the exact (height, width) shape.
        pixels = np.ascontiguousarray(mask_array, dtype=np.uint8)
        self.mmc.setSLMImage(self.label, pixels)
        self.mmc.displaySLMImage(self.label)

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
