"""
functions/mm2_slm.py (Micro-Manager 2.0 GUI stack, via pycromanager)

The SLM addressed as the Generic SLM device ('Spatial light modulator
controlled through computer graphics') of a RUNNING Micro-Manager 2.0
instance, through pycromanager's `Core` bridge - the same device
functions/mmcore_slm.py's MMCoreSLM drives in the pymmcore-plus stack.

pycromanager needs Micro-Manager's server running: in Micro-Manager,
Tools > Options > "Run pycro-manager server on port 4827".

Because Micro-Manager owns the device, a mask pushed from a Python
script stays on the SLM after the script exits, until something else
replaces it (another script, or Micro-Manager's own SLM tools).
Never writes to disk.
"""
import numpy as np


class MM2SLM:
    """
    Usage:
        from pycromanager import Core
        slm = MM2SLM(Core(), config)
        slm.show_mask(mask)
    """

    def __init__(self, core, config):
        self.core = core
        self.label = config.get("slm", "device_label", default="SLM")
        loaded = list(core.get_loaded_devices())
        if self.label not in loaded:
            raise RuntimeError(
                f"No device '{self.label}' (slm.device_label) is loaded in the running "
                f"Micro-Manager instance - add the Generic SLM device to its hardware "
                f"config, or fix slm.device_label."
            )
        width, height = core.get_slm_width(self.label), core.get_slm_height(self.label)
        cfg_width, cfg_height = config.get("slm", "resolution", default=[width, height])
        if (width, height) != (cfg_width, cfg_height):
            raise RuntimeError(
                f"slm.resolution in the config is {[cfg_width, cfg_height]}, but "
                f"Micro-Manager reports the '{self.label}' SLM device as "
                f"{width}x{height}. Make the two agree."
            )
        self.shape = (height, width)

    def show_mask(self, mask_array):
        """Push a uint8 (H, W) mask to the SLM and display it."""
        mask = np.asarray(mask_array)
        if mask.shape != self.shape:
            raise ValueError(f"Mask shape {mask.shape} != SLM shape {self.shape}")
        # A flat, row-major uint8 array is sent by the Java bridge as byte[],
        # which selects CMMCore.setSLMImage(String, byte[]) - 8-bit grey levels.
        pixels = np.ascontiguousarray(mask, dtype=np.uint8).ravel()
        self.core.set_slm_image(self.label, pixels)
        self.core.display_slm_image(self.label)
