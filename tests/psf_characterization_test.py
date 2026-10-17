"""
3D PSF characterization against a simulated microscope: pymmcore-plus
Python devices on a UniMMCore - a micromanipulator with mechanical
BACKLASH, mounted so that the depth axis is 'Current Y' (not Z), a
camera imaging a bead whose PSF widens away from focus, an SLM and a
laser. The real experiment (experiments/psf_characterization.py) runs
end to end with its prompts answered. Checks:

* only the depth property moves, and after the overshoot the stage only
  ever moves down, so every plane sees the same backlash offset;
* the stack has the requested planes, the focus is found in it, and the
  dark mean (200 frames) has been subtracted;
* the chosen phase mask is on the SLM during the scan, the stage ends at
  its starting depth (approached from above), laser off, ROI cleared;
* step sizes that aren't whole microsteps, and bad axis maps, are refused.

    python tests/psf_characterization_test.py
"""
import builtins
import json
import shutil
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import tifffile
from pymmcore_plus.experimental.unicore import (
    GenericDevice, SimpleCameraDevice, SLMDevice, UniMMCore,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import MicroscopeConfig
from functions.imaging_masks import build_imaging_mask
from functions.mmcore_micromanipulator import Micromanipulator, scan_positions

W, H = 320, 200                        # SLM
SENSOR_H, SENSOR_W = 300, 400
ROI = (40, 60, 128)                    # left channel ROI x, y, size
BEAD = (40 + 70, 60 + 55)              # bead, full-frame px
START = {"Current X": 10.0, "Current Y": 50.0, "Current Z": -5.0}
BACKLASH_UM = 0.5
FOCUS_UM = 50.0 + BACKLASH_UM / 2 - 0.75   # true focus: 0.75 um below the start (once taken up)
OFFSET, READ_NOISE, PHOTONS = 100.0, 3.0, 2e5

WORK = Path("/tmp/psf_char_test"); shutil.rmtree(WORK, ignore_errors=True)
(WORK / "config").mkdir(parents=True)
CFG = WORK / "config" / "microscope_config.json"
base = {
    "camera": {"channel_rois": {"left": {"position_px": list(ROI[:2]), "size_px": ROI[2]},
                                "right": {"position_px": [230, 60], "size_px": 128}}},
    "slm": {"device_label": "SLM", "resolution": [W, H], "flat_value": 128,
            "grey_level_2pi": 255, "refresh_rate_hz": 1000, "settle_frames": 0},
    "laser": {"device_label": "L", "power_property": "P", "state_property": "S"},
    "fourier_plane": {"radius_px": 60, "channels": {"left": {"center_px": [80, 100]},
                                                    "right": {"center_px": [235, 96]}}},
    "micromanipulator": {"device_label": "Sutter MPC", "microstep_um": 0.0625, "settle_s": 0.0,
                         "axis_properties": {"horizontal": "Current Z", "vertical": "Current X",
                                             "depth": "Current Y"}},
    "psf_characterization": {"half_range_um": 2.0, "step_um": 0.125, "overshoot_steps": 2,
                             "exposure_ms": 5, "laser_power_mW": 1.0, "guide_star_roi_px": 64,
                             "dark_frames": 200},
    "imaging_masks": {"channels": ["left", "right"], "aberration_correction_method": "spsa",
                      "edof_zernike_rad_rms": {"11": 1.5}},
    "acquisition_defaults": {"root_folder": str(WORK / "data")},
}
json.dump(base, open(CFG, "w"))
config = MicroscopeConfig(CFG)

failures = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


class SimStage(GenericDevice):
    """Reports the commanded position; the TRUE one lags by the backlash
    whenever the direction of motion reverses."""

    def __init__(self):
        super().__init__()
        self.commanded = dict(START)
        self.direction = {k: +1 for k in START}   # last move was up
        self.history = []
        for prop in START:
            self.register_property(
                prop, default_value=START[prop],
                getter=lambda self, p=prop: self.commanded[p],
                setter=lambda self, v, p=prop: self._move(p, float(v)))

    def _move(self, prop, value):
        if value != self.commanded[prop]:
            self.direction[prop] = 1 if value > self.commanded[prop] else -1
        self.commanded[prop] = value
        self.history.append((prop, value))

    def true(self, prop):
        # Moving down leaves the carriage BACKLASH/2 above the command, up -> below.
        return self.commanded[prop] - self.direction[prop] * BACKLASH_UM / 2


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
    def __init__(self, stage, laser, slm, rng):
        super().__init__(); self.stage, self.laser, self.slm, self.rng = stage, laser, slm, rng
        self._exposure, self.lit = 5.0, []
    def sensor_shape(self): return (SENSOR_H, SENSOR_W)
    def dtype(self): return np.uint16
    def get_exposure(self): return self._exposure
    def set_exposure(self, exposure): self._exposure = exposure
    def snap(self, buffer):
        h, w = buffer.shape   # full sensor; the core crops to the ROI
        yy, xx = np.mgrid[0:h, 0:w]
        signal = np.zeros((h, w))
        if self.laser.get_property_value("S") == "open":
            z = self.stage.true("Current Y")
            sigma = 1.5 * np.sqrt(1 + ((z - FOCUS_UM) / 0.6) ** 2)
            r2 = (xx - BEAD[0]) ** 2 + (yy - BEAD[1]) ** 2
            signal = PHOTONS / (2 * np.pi * sigma ** 2) * np.exp(-r2 / (2 * sigma ** 2))
            self.lit.append((self.stage.commanded["Current Y"], z, self.slm.shown.copy()))
        noisy = self.rng.poisson(signal) + self.rng.normal(OFFSET, READ_NOISE, (h, w))
        buffer[:] = np.clip(noisy, 0, 65535)
        return {}


rng = np.random.default_rng(3)
mmc = UniMMCore()
stage, slm_dev, laser = SimStage(), SimSLM(), SimLaser()
cam = SimCamera(stage, laser, slm_dev, rng)
mmc.loadPyDevice("Cam", cam); mmc.loadPyDevice("SLM", slm_dev)
mmc.loadPyDevice("L", laser); mmc.loadPyDevice("Sutter MPC", stage)
mmc.initializeAllDevices(); mmc.setCameraDevice("Cam"); mmc.setSLMDevice("SLM")

# ------------------------------------------------------------ refusals
print("refusals")
st = Micromanipulator(mmc, config)
for bad in (0.1, 0.0, -0.125):
    try:
        st.check_step(bad); check(False, f"step {bad} refused")
    except ValueError:
        check(True, f"step {bad} um refused")
st.check_step(0.0625); st.check_step(0.25)
bad_cfg = json.loads(json.dumps(base))
bad_cfg["micromanipulator"]["axis_properties"]["horizontal"] = "Current Y"
json.dump(bad_cfg, open(WORK / "bad.json", "w"))
try:
    Micromanipulator(mmc, MicroscopeConfig(WORK / "bad.json")); check(False, "dup axis refused")
except ValueError:
    check(True, "two camera axes on one stage property refused")
pos = scan_positions(50.0, 2.0, 0.125)
check(len(pos) == 33 and pos[0] == 52.0 and pos[-1] == 48.0 and np.all(np.diff(pos) < 0)
      and 50.0 in pos, "scan_positions: 33 planes, 52 -> 48, descending, centre included")

# ------------------------------------------------------------ experiment
print("experiment")
import experiments.psf_characterization as pc

answers = iter(["left", "y", "edof", "y"])  # channel, guide star ok, mask, proceed
with mock.patch.object(pc, "load_config", lambda: config), \
     mock.patch.object(pc, "load_mmcore", lambda c: mmc), \
     mock.patch.object(pc, "preview_sample_positioning", lambda *a, **k: None), \
     mock.patch.object(pc, "show_verification_frame", lambda *a, **k: None), \
     mock.patch.object(builtins, "input", lambda q="": (print(q), next(answers))[1]):
    pc.main()

moved_props = {p for p, _ in stage.history}
check(moved_props == {"Current Y"}, f"only the depth property moved ({moved_props})")

ys = [v for p, v in stage.history if p == "Current Y"]
top = 50.0 + 2.0
overshoot_idx = ys.index(top + 0.25)
scan_moves = ys[overshoot_idx + 1:overshoot_idx + 1 + 33]
check(ys[0] == top + 0.25, f"first move overshoots above the range ({ys[0]})")
check(np.allclose(scan_moves, scan_positions(50.0, 2.0, 0.125)),
      "then visits top -> bottom of the range")
check(np.all(np.diff(scan_moves) < 0), "only moving down during the scan")

lit = cam.lit
scan_lit = lit[-33:]
offsets = np.array([t - c for c, t, _ in scan_lit])
check(np.allclose(offsets, BACKLASH_UM / 2),
      f"backlash taken up on every plane (true - commanded = {np.unique(offsets.round(4))})")
edof = build_imaging_mask(config, "edof")[0]
check(all(np.array_equal(m, edof) for _, _, m in scan_lit), "EDOF mask on the SLM during the scan")
check(laser.get_property_value("S") == "closed", "laser off at the end")
check(stage.commanded["Current Y"] == 50.0 and stage.direction["Current Y"] == -1,
      "stage back at the start depth, approached from above")
check(tuple(mmc.getROI()) == (0, 0, SENSOR_W, SENSOR_H), "camera ROI cleared")

run = sorted((WORK / "data").glob("*/*_psf_characterization_*"))[-1]
stack = tifffile.imread(run / "psf_zstack.tif")
peak_yx = np.unravel_index(np.argmax(stack.max(axis=0)), stack.shape[1:])
roi = json.load(open(run / "psf_zstack_metadata.json"))["camera_roi_xywh"]
check(abs(roi[0] + peak_yx[1] - BEAD[0]) <= 1 and abs(roi[1] + peak_yx[0] - BEAD[1]) <= 1,
      f"stack centred on the bead (ROI {roi}, peak at {peak_yx})")
meta = json.load(open(run / "psf_zstack_metadata.json"))
check(stack.shape == (33, 64, 64) and stack.dtype == np.float32,
      f"stack shape {stack.shape} {stack.dtype}")
corner = stack[:, :10, :10]
check(abs(corner.mean()) < 1.0 and abs(np.median(stack)) < 2.0,
      f"dark subtracted (background mean {corner.mean():.2f}, was ~{OFFSET:.0f})")
peak_plane = int(np.argmax(stack.max(axis=(1, 2))))
expected = FOCUS_UM - BACKLASH_UM / 2 - 50.0      # focus offset in COMMANDED coordinates
check(abs(meta["z_offset_um"][peak_plane] - expected) <= 0.125,
      f"focus found at {meta['z_offset_um'][peak_plane]:+.3f} um (expected {expected:+.3f})")
check(meta["depth_axis_property"] == "Current Y" and meta["dark_frames"] == 200
      and meta["phase_mask"]["preset"] == "edof" and len(meta["z_readback_um"]) == 33,
      "metadata: depth property, 200 dark frames, mask, read-back positions")
check((run / "dark_mean.tif").exists() and (run / "slm_mask.tif").exists(),
      "dark mean and mask saved")
print(f"  run folder: {run}")

# ------------------------------------------------------------ error mid-scan
print("error mid-scan")
stage.commanded["Current Y"], stage.direction["Current Y"] = 50.0, +1
real_snap, n_lit = cam.snap, len(cam.lit)
def failing_snap(buffer):
    if len(cam.lit) >= n_lit + 1 + 10:   # verification frame + 10 planes
        raise RuntimeError("simulated camera failure")
    return real_snap(buffer)
cam.snap = failing_snap
answers = iter(["left", "y", "flat", "y"])
with mock.patch.object(pc, "load_config", lambda: config), \
     mock.patch.object(pc, "load_mmcore", lambda c: mmc), \
     mock.patch.object(pc, "preview_sample_positioning", lambda *a, **k: None), \
     mock.patch.object(pc, "show_verification_frame", lambda *a, **k: None), \
     mock.patch.object(builtins, "input", lambda q="": next(answers)):
    try:
        pc.main(); check(False, "camera failure propagates")
    except RuntimeError:
        check(True, "camera failure propagates")
cam.snap = real_snap
check(laser.get_property_value("S") == "closed", "laser off after the failure")
check(stage.commanded["Current Y"] == 50.0 and stage.direction["Current Y"] == -1,
      "stage returned to the start depth after the failure")
check(tuple(mmc.getROI()) == (0, 0, SENSOR_W, SENSOR_H), "ROI cleared after the failure")

print()
assert not failures, failures
print("PSF CHARACTERIZATION TEST PASSED")
