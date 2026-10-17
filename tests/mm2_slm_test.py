"""
The Micro-Manager 2.0 (pycromanager) mask scripts, experiments/mm2_slm_*.py,
against a fake pycromanager `Core` standing in for a running
Micro-Manager instance with a Generic SLM device. Each call the scripts
make goes through pyjavaz's own overload resolution (the Java bridge
under pycromanager) with CMMCore's real setSLMImage signatures, so the
test also checks that the mask is sent as byte[] (8-bit grey levels),
not int[] (32-bit RGB). Checks:

* every script displays exactly the mask build_imaging_mask gives for
  its preset (both channels), row-major, and records it in a run folder;
* a missing stored correction raises before the SLM is touched;
* a missing SLM device or a resolution mismatch is refused.

    python tests/mm2_slm_test.py
"""
import json
import shutil
import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
from pyjavaz.bridge import _check_method_args

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import MicroscopeConfig
from core.file_io import write_phase_mask_coefficients
from functions.imaging_masks import build_imaging_mask

W, H = 320, 200
WORK = Path("/tmp/mm2_slm_test"); shutil.rmtree(WORK, ignore_errors=True)
(WORK / "config").mkdir(parents=True)
CFG = WORK / "config" / "microscope_config.json"
json.dump({
    "slm": {"device_label": "SLM", "resolution": [W, H], "flat_value": 128, "grey_level_2pi": 255},
    "fourier_plane": {"radius_px": 60, "channels": {"left": {"center_px": [80, 100]},
                                                    "right": {"center_px": [235, 96]}}},
    "imaging_masks": {"channels": ["left", "right"], "aberration_correction_method": "spsa",
                      "edof_zernike_rad_rms": {"11": 1.5}},
    "acquisition_defaults": {"root_folder": str(WORK / "data")},
}, open(CFG, "w"))
config = MicroscopeConfig(CFG)

failures = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


# CMMCore's setSLMImage overloads, as the bridge sees them.
SET_SLM_IMAGE = [{"arguments": ["java.lang.String", "byte[]"]},
                 {"arguments": ["java.lang.String", "int[]"]}]


class FakeCore:
    """pycromanager Core of a running MM 2.0 with a Generic SLM device."""

    def __init__(self, devices=("Core", "Camera", "SLM"), size=(W, H)):
        self.devices, self.size = list(devices), size
        self.loaded = self.shown = None
        self.overloads = []

    def get_loaded_devices(self): return self.devices
    def get_slm_width(self, label): return self.size[0]
    def get_slm_height(self, label): return self.size[1]

    def set_slm_image(self, label, pixels):
        spec, _ = _check_method_args(SET_SLM_IMAGE, [label, pixels])
        self.overloads.append(spec["arguments"][1])
        # What Java receives: the array's raw bytes, row-major.
        self.loaded = np.frombuffer(pixels.tobytes(), np.uint8).reshape(self.size[1], self.size[0])

    def display_slm_image(self, label): self.shown = self.loaded


core = FakeCore()
sys.modules["pycromanager"] = types.SimpleNamespace(Core=lambda: core)

import experiments.mm2_slm_aberration_correction as ao
import experiments.mm2_slm_aberration_correction_edof as ao_edof
import experiments.mm2_slm_edof as edof
import experiments.mm2_slm_flat as flat

scripts = {flat: "flat", edof: "edof", ao: "aberration_correction",
           ao_edof: "aberration_correction_edof"}


def run(module):
    with mock.patch.object(module, "load_config", lambda: config):
        return module.main()


print("missing stored correction")
for module in (ao, ao_edof):
    core.shown = None
    try:
        run(module); check(False, f"{module.__name__} raises")
    except ValueError:
        check(core.shown is None, f"{module.__name__}: raises before touching the SLM")

for ch, c in {"left": {"5": 0.4, "7": -0.3}, "right": {"6": -0.5, "11": 0.2}}.items():
    write_phase_mask_coefficients(config.phase_masks_path, "spsa", ch, {"coefficients_rad_rms": c})

print("scripts")
for module, preset in scripts.items():
    run(module)
    name = module.__name__.split(".")[-1]
    expected = build_imaging_mask(config, preset)[0]
    check(core.shown is not None and np.array_equal(core.shown, expected),
          f"{name}: '{preset}' mask displayed")
    check(core.overloads[-1] == "byte[]", f"{name}: sent as byte[] (8-bit)")
    rec = sorted((WORK / "data").glob(f"*/*_{name}_*"))
    check(rec and json.load(open(rec[-1] / "slm_mask_metadata.json"))["preset"] == preset,
          f"{name}: recorded in a run folder")
check(not np.array_equal(build_imaging_mask(config, "edof")[0].T.ravel(),
                         build_imaging_mask(config, "edof")[0].ravel()),
      "(the row-major check above is meaningful: the mask isn't transpose-symmetric)")

print("refusals")
for bad, why in [(FakeCore(devices=("Core", "Camera")), "no SLM device"),
                 (FakeCore(size=(W, H + 1)), "resolution mismatch")]:
    try:
        with mock.patch.object(flat, "Core", lambda bad=bad: bad):
            run(flat)
        check(False, f"{why} refused")
    except RuntimeError as exc:
        check(bad.shown is None, f"{why} refused ({str(exc)[:60]}...)")

print()
assert not failures, failures
print("MM2 SLM TEST PASSED")
