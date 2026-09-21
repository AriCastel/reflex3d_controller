"""
functions/slm.py (legacy, pycromanager stack)

General-purpose SLM control: phase mask generation and fullscreen
display on the SLM's secondary monitor. Never writes to disk.

The SLM is addressed as a plain secondary display (not through
Micro-Manager), via a persistent, borderless tkinter window
positioned with screeninfo. Keeping the window persistent (rather
than recreating it per-frame) means the mask can be swapped out with
`show_mask()` mid-experiment without any visible flicker/reposition.

Doesn't touch `core`/pycromanager at all, so it's reused as-is by the
pymmcore-plus `*_mda` experiments too.
"""
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from screeninfo import get_monitors


def flat_mask(config, value=None):
    """
    Build a neutral, flat phase mask: a uint8 array of constant grey
    level, sized to the SLM's native resolution.
    """
    width, height = config.get("slm", "resolution", default=[1920, 1080])
    if value is None:
        value = config.get("slm", "flat_value", default=128)
    return np.full((height, width), value, dtype=np.uint8)


class SLMDisplay:
    """
    Persistent fullscreen window on the SLM's monitor.

    Usage:
        with SLMDisplay(config) as slm:
            slm.show_mask(flat_mask(config))
            ... run the experiment while the mask stays up ...
        # window is closed automatically on exit
    """

    def __init__(self, config):
        self.config = config
        self.monitor_index = config.get("slm", "monitor_index", default=1)
        self._root = None
        self._label = None
        self._photo = None  # keep a reference alive against garbage collection

    def __enter__(self):
        monitors = get_monitors()
        if self.monitor_index >= len(monitors):
            raise RuntimeError(
                f"slm.monitor_index={self.monitor_index} in the config, but "
                f"only {len(monitors)} monitor(s) were detected by screeninfo."
            )
        mon = monitors[self.monitor_index]

        self._root = tk.Tk()
        self._root.overrideredirect(True)  # borderless, no window chrome
        self._root.geometry(f"{mon.width}x{mon.height}+{mon.x}+{mon.y}")
        self._label = tk.Label(self._root, bd=0, highlightthickness=0)
        self._label.pack(fill=tk.BOTH, expand=True)
        self._root.update()
        return self

    def show_mask(self, mask_array):
        """Push a new uint8 (H, W) mask array to the SLM window."""
        img = Image.fromarray(mask_array, mode="L")
        self._photo = ImageTk.PhotoImage(img)
        self._label.configure(image=self._photo)
        self._root.update()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._root is not None:
            self._root.destroy()
