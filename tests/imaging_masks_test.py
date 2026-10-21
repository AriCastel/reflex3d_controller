"""
Phase-mask store + imaging masks against a simulated two-channel
microscope (pymmcore-plus Python devices on a UniMMCore: camera, SLM,
laser). Checks:

* config/phase_masks.json writes (method -> channel), backups, and
  that storing one channel keeps the other;
* each preset's mask, decoded back to phase, is the expected Zernike
  sum in each channel's pupil and flat everywhere else (correction +
  EDOF adds the two term by term);
* a missing stored correction raises before anything is shown;
* experiments/load_phase_mask.py finds the core running in this process
  (the napari GUI's) and puts each preset's mask on the SLM, and refuses
  with a clear error when there is no core to be found;
(The timelapse's phase mask is covered by tests/timelapse_test.py.)

    python tests/imaging_masks_test.py
"""
import json
import shutil
import sys
from pathlib import Path
from unittest import mock

import numpy as np
from pymmcore_plus.experimental.unicore import (
    GenericDevice, SimpleCameraDevice, SLMDevice, UniMMCore,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import MicroscopeConfig, load_phase_masks
from core.file_io import load_ao_metadata, write_phase_mask_coefficients
from functions.adaptive_optics import AO_METHOD, ZernikePupil, phase_mask_entry
from functions.imaging_masks import PRESETS, build_imaging_mask, describe_imaging_mask
from functions.zernike import noll_zernike

W, H, R = 320, 200, 60
CENTERS = {"left": (80.0, 100.0), "right": (235.5, 96.0)}
FLAT, G2PI = 128, 255
EDOF = {11: 1.5}
CORRECTION = {
    "left": {5: 0.4, 7: -0.3, 11: 0.2},
    "right": {6: -0.5, 8: 0.25, 11: -0.1, 12: 0.15},
}

WORK = Path("/tmp/imaging_masks_test")
shutil.rmtree(WORK, ignore_errors=True)
(WORK / "config").mkdir(parents=True)
CFG = WORK / "config" / "microscope_config.json"
json.dump({
    "camera": {"channel_rois": {}},
    "slm": {"device_label": "SLM", "resolution": [W, H], "flat_value": FLAT,
            "grey_level_2pi": G2PI, "refresh_rate_hz": 1000, "settle_frames": 0},
    "laser": {"device_label": "L", "power_property": "P", "state_property": "S"},
    "fourier_plane": {"radius_px": R, "channels": {
        ch: {"center_px": list(c)} for ch, c in CENTERS.items()}},
    "acquisition_defaults": {"root_folder": str(WORK / "data")},
    "imaging_masks": {"channels": ["left", "right"], "aberration_correction_method": "spsa",
                      "edof_zernike_rad_rms": {str(j): c for j, c in EDOF.items()}},
}, open(CFG, "w"))
config = MicroscopeConfig(CFG)

failures = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def decode(mask, channel):
    """Phase (rad, wrapped) in a channel's pupil, and the pupil's grid."""
    cx, cy = CENTERS[channel]
    yy, xx = np.mgrid[0:H, 0:W]
    rho = np.hypot(xx - cx, yy - cy) / R
    theta = np.arctan2(yy - cy, xx - cx)
    inside = rho <= 1
    phase = mask.astype(float) / G2PI * 2 * np.pi - FLAT / G2PI * 2 * np.pi
    return phase, rho, theta, inside


def expected_phase(terms, rho, theta):
    return sum(c * noll_zernike(j, rho, theta) for j, c in terms.items()) if terms else 0 * rho


def check_mask(mask, terms_by_channel, label):
    outside = np.ones((H, W), bool)
    for ch, terms in terms_by_channel.items():
        phase, rho, theta, inside = decode(mask, ch)
        err = np.angle(np.exp(1j * (phase - expected_phase(terms, rho, theta))))[inside]
        check(np.abs(err).max() < np.pi / G2PI + 1e-3,
              f"{label}: '{ch}' pupil phase matches (max err {np.abs(err).max():.4f} rad)")
        outside &= ~inside
    check(np.all(mask[outside] == FLAT), f"{label}: flat outside the pupils")


# ------------------------------------------------------------------ store
print("phase_masks.json store")
check(load_phase_masks(config) == {}, "missing file reads as empty")
try:
    build_imaging_mask(config, "aberration_correction")
    check(False, "missing correction raises")
except ValueError as exc:
    check("No 'spsa' aberration correction stored for channel 'left'" in str(exc),
          "missing correction raises a clear error")

# Store the left channel from a run's metadata, as aberration_correction.py does.
run = WORK / "ao_run_left"; run.mkdir()
meta_left = {"method": AO_METHOD, "channel": "left", "finished_at": "2026-10-14 10:00:00",
             "coefficients_rad_rms": {str(j): c for j, c in CORRECTION["left"].items()},
             "metric": {"name": "power_weighted_second_moment"},
             "metric_original_end": 3.0, "metric_corrected_end": 1.0,
             "power_ratio_corrected_over_original": 1.1,
             "fourier_center_px": list(CENTERS["left"]), "pupil_radius_px": R}
json.dump(meta_left, open(run / "ao_correction_metadata.json", "w"))
meta, folder = load_ao_metadata(run)
_, backup = write_phase_mask_coefficients(config.phase_masks_path, AO_METHOD, "left",
                                          phase_mask_entry(meta, folder))
check(backup is None, "first write makes no backup")

try:
    build_imaging_mask(config, "aberration_correction")
    check(False, "half-stored correction raises")
except ValueError as exc:
    check("'right'" in str(exc), "only the right channel missing -> raises for 'right'")

# A wrong right-channel entry, then the right one (replacing it).
write_phase_mask_coefficients(config.phase_masks_path, AO_METHOD, "right",
                              {"coefficients_rad_rms": {"5": 9.0}})
_, backup = write_phase_mask_coefficients(
    config.phase_masks_path, AO_METHOD, "right",
    {"coefficients_rad_rms": {str(j): c for j, c in CORRECTION["right"].items()}})
store = load_phase_masks(config)
check(backup is not None and backup.exists(), "rewrite backs the previous file up")
check(json.load(open(backup))[AO_METHOD]["right"]["coefficients_rad_rms"] == {"5": 9.0},
      "backup holds the previous entry")
check(store[AO_METHOD]["right"]["coefficients_rad_rms"].get("5") is None,
      "an entry is replaced, not merged")
check(store[AO_METHOD]["left"]["run"] == str(run) and
      store[AO_METHOD]["left"]["measured_on"] == "2026-10-14 10:00:00",
      "storing 'right' keeps 'left' (with its provenance)")

# ------------------------------------------------------------------ masks
print("presets")
expected = {
    "flat": {ch: {} for ch in CENTERS},
    "aberration_correction": CORRECTION,
    "edof": {ch: EDOF for ch in CENTERS},
    "aberration_correction_edof": {
        ch: {j: CORRECTION[ch].get(j, 0) + EDOF.get(j, 0) for j in set(CORRECTION[ch]) | set(EDOF)}
        for ch in CENTERS},
}
masks = {}
for preset in PRESETS:
    mask, info = build_imaging_mask(config, preset)
    masks[preset] = mask
    check(mask.shape == (H, W) and mask.dtype == np.uint8, f"{preset}: full-SLM uint8")
    check_mask(mask, expected[preset], preset)
    check(not info["warnings"], f"{preset}: no overlap warning")
print(describe_imaging_mask(build_imaging_mask(config, "aberration_correction_edof")[1]))

mask, _ = build_imaging_mask(config, "edof", channels=["right"])
check_mask(mask, {"right": EDOF}, "edof on 'right' only")
_, rho, _, inside_left = decode(mask, "left")
check(np.all(mask[inside_left] == FLAT), "edof on 'right' only: left pupil flat")

# The stored correction gives the same mask as the AO code's own pupil.
pupil = ZernikePupil(config, "left", sorted(CORRECTION["left"]))
ref = pupil.mask([CORRECTION["left"][j] for j in sorted(CORRECTION["left"])])
_, _, _, inside = decode(ref, "left")
check(np.array_equal(ref[inside], masks["aberration_correction"][inside]),
      "stored correction == ZernikePupil.mask of the same coefficients")

# ------------------------------------------------------------------ scripts
print("mask scripts in a running GUI process")


class SimSLM(SLMDevice):
    def __init__(self):
        super().__init__(); self.loaded = self.shown = np.zeros((H, W), np.uint8)
    def shape(self): return (H, W)
    def dtype(self): return np.uint8
    def set_image(self, pixels): self.loaded = np.array(pixels)
    def display_image(self): self.shown = self.loaded
    def set_exposure(self, interval_ms): pass
    def get_exposure(self): return 0.0


class SimLaser(GenericDevice):
    def __init__(self):
        super().__init__()
        self.register_property("P", default_value=0.0)
        self.register_property("S", default_value="closed", allowed_values=["open", "closed"])


class SimCamera(SimpleCameraDevice):
    def __init__(self, laser):
        super().__init__(); self.laser, self._exposure, self.lit = laser, 10.0, 0
    def sensor_shape(self): return (64, 64)
    def dtype(self): return np.uint16
    def get_exposure(self): return self._exposure
    def set_exposure(self, exposure): self._exposure = exposure
    def snap(self, buffer):
        buffer[:] = 100 + 1000 * (self.laser.get_property_value("S") == "open")
        self.lit += self.laser.get_property_value("S") == "open"
        return {}


import experiments.load_phase_mask as lpm

# No Micro-Manager 2.0 either: block pycromanager (only that key - patch.dict on
# sys.modules would also drop every module imported meanwhile, e.g. Qt's).
_saved_pycromanager = sys.modules.get("pycromanager")
sys.modules["pycromanager"] = None
with mock.patch.object(lpm, "load_config", lambda: config):
    try:
        lpm.main("flat")
        check(False, "load_phase_mask: refuses without a core")
    except RuntimeError as exc:
        check("No Micro-Manager core with the SLM found" in str(exc),
              "load_phase_mask: refuses without a core")

# The "GUI": a core in this process with the SLM loaded (UniMMCore()
# becomes CMMCorePlus.instance(), as load_mmcore's core would).
mmc = UniMMCore()
slm_dev, laser = SimSLM(), SimLaser()
mmc.loadPyDevice("Cam", SimCamera(laser)); mmc.loadPyDevice("SLM", slm_dev)
mmc.loadPyDevice("L", laser)
mmc.initializeAllDevices(); mmc.setCameraDevice("Cam"); mmc.setSLMDevice("SLM")

for preset in PRESETS:
    with mock.patch.object(lpm, "load_config", lambda: config):  # pycromanager still blocked
        lpm.main(preset)
    check(np.array_equal(slm_dev.shown, masks[preset]),
          f"load_phase_mask '{preset}': mask on the SLM")
    check(not list((WORK / "data").glob("*/*_load_phase_mask_*")),
          f"load_phase_mask '{preset}': no run folder written")

if _saved_pycromanager is None:
    del sys.modules["pycromanager"]
else:
    sys.modules["pycromanager"] = _saved_pycromanager

print()
assert not failures, failures
print("IMAGING MASKS TEST PASSED")
