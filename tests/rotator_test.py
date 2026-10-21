"""
The PR50CC rotation stage: functions/mmcore_rotator.py and the napari
GUI's 'Rotator' panel, against a simulated stage (a pymmcore-plus Python
device with the real device's two properties: 'find_home', set to the
command string "FIND HOME NOW", and 'X Axis', the absolute position in
milli-degrees). The simulated stage rejects anything but absolute
positions and records every value written to it. Checks:

* initialize writes the home command; the angle is read as degrees
  (90000 -> 90.0, -90000 -> -90.0);
* a relative rotation is the position read from the stage plus the
  change, written as one absolute value in milli-degrees (a 3 degree turn
  from 12.5 degrees writes 15500), also for negative and sub-degree steps;
* min_deg / max_deg refuse a move without touching the stage;
* the panel's coarse and fine step sizes can be changed on the fly and
  each click uses the value it has then; Initialize asks for
  confirmation first; the panel is locked while a move is under way;
  a failing move is reported without taking the GUI down;
* experiments/imaging_gui.py adds the panel when the stage is in the
  hardware config (and not otherwise);
* nothing is written to disk by any of it (no run folder, no log).

    xvfb-run -a python tests/rotator_test.py
"""
import json
import shutil
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np
from pymmcore_plus.experimental.unicore import (
    GenericDevice, SimpleCameraDevice, SLMDevice, UniMMCore,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules["pycromanager"] = None

from core.config import MicroscopeConfig
from functions.mmcore_rotator import Rotator

WORK = Path("/tmp/rotator_test"); shutil.rmtree(WORK, ignore_errors=True)
(WORK / "config").mkdir(parents=True)
CFG = WORK / "config" / "microscope_config.json"
ROTATOR_CFG = {"device_label": "PR50CC", "position_property": "X Axis",
               "home_property": "find_home", "home_command": "FIND HOME NOW",
               "units_per_degree": 1000, "settle_s": 0.0,
               "coarse_step_deg": 45.0, "fine_step_deg": 10.0}


def write_config(rotator=ROTATOR_CFG, name="microscope_config.json"):
    path = WORK / "config" / name
    json.dump({
        "slm": {"device_label": "SLM", "resolution": [64, 48], "flat_value": 128,
                "grey_level_2pi": 255, "refresh_rate_hz": 1000, "settle_frames": 0},
        "laser": {"device_label": "L", "power_property": "P", "state_property": "S",
                  "default_power_mW": 20.0, "max_power_mW": 200},
        "fourier_plane": {"radius_px": 10, "channels": {"left": {"center_px": [16, 24]},
                                                        "right": {"center_px": [48, 24]}}},
        "imaging_masks": {"channels": ["left", "right"]},
        "acquisition_defaults": {"root_folder": str(WORK / "data")},
        **({"rotator": rotator} if rotator is not None else {}),
    }, open(path, "w"))
    return MicroscopeConfig(path)


config = write_config()
failures = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


class SimRotator(GenericDevice):
    """Absolute positioning only. Homing takes `home_delay_s` and goes to 0."""
    def __init__(self, home_delay_s=0.0, move_delay_s=0.0):
        super().__init__()
        self.position, self.writes, self.homed = 0, [], 0
        self.home_delay_s, self.move_delay_s, self.fail = home_delay_s, move_delay_s, False
        self.register_property("X Axis", default_value=0, property_type=int,
                               getter=lambda dev: dev.position, setter=SimRotator._set_position)
        self.register_property("find_home", default_value="",
                               allowed_values=["", "FIND HOME NOW"],
                               setter=SimRotator._home)

    def _set_position(self, value):
        if self.fail:
            raise RuntimeError("stage fault")
        time.sleep(self.move_delay_s)
        self.writes.append(int(value)); self.position = int(value)

    def _home(self, value):
        if value == "FIND HOME NOW":
            time.sleep(self.home_delay_s)
            self.homed += 1; self.position = 0


class SimSLM(SLMDevice):
    def __init__(self):
        super().__init__(); self.shown = np.zeros((48, 64), np.uint8)
    def shape(self): return (48, 64)
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
    def __init__(self): super().__init__(); self._e = 5.0
    def sensor_shape(self): return (48, 64)
    def dtype(self): return np.uint16
    def get_exposure(self): return self._e
    def set_exposure(self, e): self._e = e
    def snap(self, buffer): buffer[:] = 100; return {}


def make_core(rotator=True, **rot_kwargs):
    mmc = UniMMCore()
    mmc.loadPyDevice("Cam", SimCamera())
    slm, laser = SimSLM(), SimLaser()
    mmc.loadPyDevice("SLM", slm); mmc.loadPyDevice("L", laser)
    rot = None
    if rotator:
        rot = SimRotator(**rot_kwargs); mmc.loadPyDevice("PR50CC", rot)
    mmc.initializeAllDevices(); mmc.setCameraDevice("Cam"); mmc.setSLMDevice("SLM")
    return mmc, rot


def data_files():
    return sorted(p for p in WORK.rglob("*") if p.is_file() and p.parent.name != "config")


# ------------------------------------------------------------ logic
print("Rotator")
mmc, rot = make_core()
r = Rotator(mmc, config)
check(r.coarse_step_deg == 45.0 and r.fine_step_deg == 10.0, "step defaults from the config")

rot.position = 90000
check(r.position_deg() == 90.0, "90000 reads as 90 degrees")
rot.position = -90000
check(r.position_deg() == -90.0, "-90000 reads as -90 degrees")

angle = r.initialize()
check(rot.homed == 1 and angle == 0.0, "initialize sets find_home = 'FIND HOME NOW'")

rot.position, rot.writes = 12500, []
check(r.rotate_by(3.0) == 15.5 and rot.writes == [15500],
      f"+3 deg from 12.5 writes the absolute 15500 ({rot.writes})")
check(r.rotate_by(-3.0) == 12.5 and rot.writes[-1] == 12500, "-3 deg goes back to 12500")
check(r.rotate_by(0.5) == 13.0 and rot.writes[-1] == 13000, "0.5 deg fine step: +500")
rot.position = 0
r.rotate_by(-90.0)
check(rot.writes[-1] == -90000 and r.position_deg() == -90.0, "-90 deg from 0 writes -90000")
rot.position = 10000
r.rotate_by(0.0015)   # finer than the stage resolution: rounds to a whole milli-degree
check(rot.writes[-1] == 10002, f"sub-milli-degree request rounds ({rot.writes[-1]})")

limited = Rotator(mmc, write_config({**ROTATOR_CFG, "min_deg": -100, "max_deg": 100}, "lim.json"))
rot.position, rot.writes = 95000, []
try:
    limited.rotate_by(10.0); check(False, "move past max_deg refused")
except ValueError:
    check(rot.writes == [] and rot.position == 95000, "move past max_deg refused, stage untouched")
limited.rotate_by(5.0)
check(rot.position == 100000, "move up to max_deg allowed")

# --------------------------------------------------------------- panel
print("napari 'Rotator' panel")
import napari
from qtpy.QtWidgets import QApplication, QDoubleSpinBox, QLabel, QPushButton

from functions.napari_preview import add_rotator_panel


def pump(seconds, until=None):
    app = QApplication.instance()
    t0 = time.time()
    while time.time() - t0 < seconds and not (until and until()):
        app.processEvents(); time.sleep(0.01)


def button(panel, fragment):
    return next(b for b in panel.findChildren(QPushButton) if fragment in b.text())


def label_with(panel, fragment):
    return next((lab.text() for lab in panel.findChildren(QLabel) if fragment in lab.text()), "")


files_before = data_files()
mmc, rot = make_core(move_delay_s=0.3, home_delay_s=0.3)
r = Rotator(mmc, config)
rot.position = 20000
viewer = napari.Viewer(show=False)
asked = []
answer = [False]
panel = add_rotator_panel(viewer, r.initialize, r.rotate_by, r.position_deg,
                          coarse_deg=r.coarse_step_deg, fine_deg=r.fine_step_deg,
                          confirm_initialize=lambda: asked.append(1) or answer[0])
pump(5, until=lambda: "20.000" in label_with(panel, "Angle"))
check("20.000" in label_with(panel, "Angle"), f"shows the angle at start ({label_with(panel, 'Angle')})")
coarse_spin, fine_spin = panel.findChildren(QDoubleSpinBox)
check((coarse_spin.value(), fine_spin.value()) == (45.0, 10.0), "steps start at the config values")
texts = sorted(b.text() for b in panel.findChildren(QPushButton))
check(any("45°" in t and "↺" in t for t in texts) and any("10°" in t and "↻" in t for t in texts),
      f"buttons show the steps {texts}")

# Initialize: declined, then confirmed.
button(panel, "Initialize").click(); pump(0.3)
check(asked == [1] and rot.homed == 0, "Initialize asks first; declined -> nothing happens")
answer[0] = True
button(panel, "Initialize").click()
check(not button(panel, "Initialize").isEnabled() and not button(panel, "45").isEnabled(),
      "panel locked while the stage moves")
pump(5, until=lambda: rot.homed == 1 and button(panel, "Initialize").isEnabled())
check(rot.homed == 1 and "0.000" in label_with(panel, "Angle"), "Initialize homes the stage")

rot.writes = []
button(panel, "+45").click()
pump(5, until=lambda: len(rot.writes) == 1 and button(panel, "+45").isEnabled())
button(panel, "+10").click()
pump(5, until=lambda: len(rot.writes) == 2 and button(panel, "+10").isEnabled())
button(panel, "↺ −45").click()
pump(5, until=lambda: len(rot.writes) == 3 and button(panel, "+10").isEnabled())
button(panel, "↺ −10").click()
pump(5, until=lambda: len(rot.writes) == 4 and button(panel, "+10").isEnabled())
check(rot.writes == [45000, 55000, 10000, 0],
      f"coarse/fine, clockwise/counter-clockwise: {rot.writes}")
check("0.000" in label_with(panel, "Angle"), f"angle shown: {label_with(panel, 'Angle')}")

# New step sizes for the next data run - no restart.
coarse_spin.setValue(10.0); fine_spin.setValue(0.5)
pump(0.2)
check(button(panel, "↺ −10°").isEnabled() and button(panel, "+0.5°").isEnabled(),
      "button labels follow the new steps")
rot.writes = []
button(panel, "+10°").click()
pump(5, until=lambda: len(rot.writes) == 1 and button(panel, "+0.5").isEnabled())
button(panel, "↺ −0.5").click()
pump(5, until=lambda: len(rot.writes) == 2 and button(panel, "+0.5").isEnabled())
check(rot.writes == [10000, 9500], f"new coarse 10 deg and fine 0.5 deg used: {rot.writes}")

# A failing move is reported and the panel is usable again.
rot.fail = True
button(panel, "+0.5").click()
pump(5, until=lambda: "FAILED" in label_with(panel, "FAILED") and button(panel, "+0.5").isEnabled())
check("FAILED" in label_with(panel, "FAILED") and button(panel, "+0.5").isEnabled(),
      f"failure reported, panel unlocked ({label_with(panel, 'FAILED')[:60]}...)")
rot.fail = False
viewer.close()
pump(0.2)
check(data_files() == files_before, "no files written (no run folder, no log)")

# -------------------------------------------------- imaging_gui wiring
print("imaging_gui")
import experiments.imaging_gui as gui

docks = {}


def build(with_rotator, cfg):
    mmc, _ = make_core(rotator=with_rotator)

    def inspect():
        viewer = napari.current_viewer()
        pump(0.3)
        docks[with_rotator] = list(viewer.window._dock_widgets)
        viewer.close()

    with mock.patch.object(gui, "load_config", lambda: cfg), \
         mock.patch.object(gui, "load_mmcore", lambda c: mmc), \
         mock.patch.object(gui.napari, "run", inspect):
        gui.main()


build(True, config)
check("Rotator" in docks[True], f"panel added when the stage is loaded {docks[True]}")
with mock.patch.object(gui.warnings, "warn") as warn:
    build(False, config)
check("Rotator" not in docks[False] and any("PR50CC" in str(c) for c in warn.call_args_list),
      "no panel (with a warning naming the device) when the stage isn't in the hardware config")
check(data_files() == files_before, "imaging_gui startup wrote nothing for the rotator")

print()
assert not failures, failures
print("ROTATOR TEST PASSED")
