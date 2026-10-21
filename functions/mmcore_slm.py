"""
functions/mmcore_slm.py

The SLM addressed as a Micro-Manager device - the "Generic SLM: Spatial
light modulator controlled through computer graphics" device - which
owns the fullscreen window on the SLM's monitor. This module only
pushes uint8 pixel buffers to it. Never writes to disk.

The same device can be reached two ways, and `MMCoreSLM` takes either:

* a pymmcore-plus `CMMCorePlus` in this process (the napari GUI, the
  headless experiments), camelCase API;
* pycromanager's `Core`, a bridge to a separately running Micro-Manager
  2.0 instance, snake_case API.

`find_slm_core` picks the right one for a script that only needs to put
a mask on the SLM (experiments/load_phase_mask.py).
"""
import numpy as np

#: Where find_slm_core may look for the core driving the SLM.
BACKENDS = ("napari", "micromanager")


def slm_settle_seconds(config):
    """
    How long to wait after pushing a mask before snapping.

    The SLM runs at a fixed refresh rate, so a frame snapped too early
    shows the PREVIOUS mask. Waiting a few frame periods covers both the
    refresh latency and liquid-crystal relaxation.
    """
    rate = config.get("slm", "refresh_rate_hz", default=30)
    frames = config.get("slm", "settle_frames", default=3)
    return float(frames) / float(rate)


class MMCoreSLM:
    """
    Thin wrapper around Micro-Manager's Generic SLM device, addressed
    through a CMMCorePlus instance or a pycromanager Core.

    Usage:
        with MMCoreSLM(mmc, config) as slm:
            slm.show_mask(flat_mask(config))
            ... run the experiment while the mask stays up ...
    """

    def __init__(self, mmc, config):
        self.mmc = mmc
        self.config = config
        self.label = config.get("slm", "device_label", default="SLM")
        # pycromanager's Core exposes MMCore's methods in snake_case.
        self._pycromanager = not hasattr(mmc, "setSLMImage")

    def _call(self, camel, snake, *args):
        return getattr(self.mmc, snake if self._pycromanager else camel)(*args)

    def __enter__(self):
        loaded = list(self._call("getLoadedDevices", "get_loaded_devices"))
        if self.label not in loaded:
            raise RuntimeError(
                f"No device '{self.label}' (slm.device_label) is loaded in Micro-Manager - "
                f"add the Generic SLM device to the hardware config, or fix slm.device_label."
            )
        core_width = self._call("getSLMWidth", "get_slm_width", self.label)
        core_height = self._call("getSLMHeight", "get_slm_height", self.label)
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
        self.shape = (core_height, core_width)
        return self

    def show_mask(self, mask_array):
        """Push a new uint8 (H, W) mask array to the SLM and display it."""
        pixels = np.ascontiguousarray(mask_array, dtype=np.uint8)
        if self._pycromanager:
            # The Java bridge sends a flat uint8 array as byte[], which selects
            # setSLMImage(String, byte[]) - 8-bit grey levels, row-major.
            pixels = pixels.ravel()
        # Otherwise kept 2D: the C++ core only checks the byte count, but
        # Python (UniMMCore) SLM devices require the exact (height, width) shape.
        self._call("setSLMImage", "set_slm_image", self.label, pixels)
        self._call("displaySLMImage", "display_slm_image", self.label)

    def acquisition_running(self):
        """True while a pymmcore-plus MDA runs (not detectable through pycromanager)."""
        mda = getattr(self.mmc, "mda", None)
        return bool(mda is not None and mda.is_running())

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def find_slm_core(config, backend=None):
    """
    The core driving the SLM, for a script that loads a mask onto it:

    * "napari": the CMMCorePlus of a napari GUI running in THIS process
      (napari-micromanager's core, or the CMMCorePlus singleton that
      experiments/imaging_gui.py creates) - the script is then run from
      the GUI's panel or console;
    * "micromanager": a running Micro-Manager 2.0 instance, through
      pycromanager - the script is then run from a terminal.

    backend=None tries "napari" first, then "micromanager". Micro-Manager
    hardware can only be open in one process, so a terminal script can't
    reach the napari GUI's hardware - only a Micro-Manager 2.0 instance.
    """
    if backend not in (None, *BACKENDS):
        raise ValueError(f"backend must be None or one of {BACKENDS}, got '{backend}'")
    label = config.get("slm", "device_label", default="SLM")
    problems = []

    if backend in (None, "napari"):
        try:
            from pymmcore_plus import CMMCorePlus
            candidates = []
            try:
                from napari_micromanager import get_core
                candidates.append(get_core())
            except Exception:  # not installed, an older version, or no GUI open
                pass
            candidates.append(CMMCorePlus.instance())
            for mmc in candidates:
                if label in mmc.getLoadedDevices():
                    return mmc
            problems.append(f"no pymmcore-plus core in this process has '{label}' loaded")
        except ImportError as exc:
            problems.append(f"pymmcore-plus unavailable ({exc})")

    if backend in (None, "micromanager"):
        try:
            from pycromanager import Core
            return Core()
        except Exception as exc:
            problems.append(f"couldn't connect to Micro-Manager 2.0 through pycromanager ({exc})")

    raise RuntimeError(
        "No Micro-Manager core with the SLM found: " + "; ".join(problems) + ". "
        "From the napari GUI (python -m experiments.imaging_gui), use its 'SLM phase mask' "
        "panel or %run in its console. With Micro-Manager 2.0 open, run from a terminal, "
        "with Tools > Options > 'Run pycro-manager server on port 4827' enabled."
    )
