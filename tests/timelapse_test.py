"""
experiments/timelapse.py - the continuous timelapse - against a simulated
microscope (pymmcore-plus Python devices on a UniMMCore: a camera that
writes its frame number into every pixel, an SLM, a laser), both headless
and from the napari GUI's 'Timelapse' panel. Checks:

* <root>/YYYYMMDD/YYYYMMDD_<name>_N/<name>.tif holds every frame, in
  order, as an ImageJ TYX stack, with <name>_metadata.json (settings,
  frames acquired, measured sampling rate, timestamps, ROI, mask);
* only a time loop is run (no stage involved), the laser is on during the
  run and off after it, and the run is continuous (sequenced);
* headless: PHASE_MASK is shown for the run and saved with it; a missing
  stored correction or name stops it before anything is created;
* stopping early (closing the headless preview, or the GUI's Stop) keeps
  the frames acquired so far and shrinks the file to them;
* the GUI panel shows the sampling rate for the exposure, runs with the
  mask last loaded from the 'SLM phase mask' panel, and reports the result.

    xvfb-run -a python tests/timelapse_test.py
"""
import json
import shutil
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np
import tifffile
from pymmcore_plus.experimental.unicore import (
    GenericDevice, SimpleCameraDevice, SLMDevice, UniMMCore,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules["pycromanager"] = None  # nothing here may need Micro-Manager 2.0

from core.config import MicroscopeConfig
from core.file_io import write_phase_mask_coefficients
from functions.imaging_masks import build_imaging_mask

W, H = 320, 200            # SLM
CAM_H, CAM_W = 48, 64
WORK = Path("/tmp/timelapse_test"); shutil.rmtree(WORK, ignore_errors=True)
(WORK / "config").mkdir(parents=True)
CFG = WORK / "config" / "microscope_config.json"
json.dump({
    "slm": {"device_label": "SLM", "resolution": [W, H], "flat_value": 128,
            "grey_level_2pi": 255, "refresh_rate_hz": 1000, "settle_frames": 0},
    "laser": {"device_label": "L", "power_property": "P", "state_property": "S",
              "default_power_mW": 20.0, "max_power_mW": 200},
    "fourier_plane": {"radius_px": 60, "channels": {"left": {"center_px": [80, 100]},
                                                    "right": {"center_px": [235, 96]}}},
    "imaging_masks": {"channels": ["left", "right"], "aberration_correction_method": "spsa",
                      "edof_zernike_rad_rms": {"11": 1.5}},
    "acquisition_defaults": {"root_folder": str(WORK / "data"),
                             "timelapse": {"num_frames": 30, "exposure_ms": 5}},
}, open(CFG, "w"))
config = MicroscopeConfig(CFG)
for ch, c in {"left": {"5": 0.4}, "right": {"6": -0.3}}.items():
    write_phase_mask_coefficients(config.phase_masks_path, "spsa", ch, {"coefficients_rad_rms": c})

failures = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


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
    """Pixel value = (frame number % 16) * 4096 + its position on the sensor
    (row * CAM_W + col), so both frame order and the crop can be checked.
    Logs laser state, power, the SLM mask, and whether the run's TIFF already
    exists on disk while frames are still coming in."""
    def __init__(self, laser, slm):
        super().__init__(); self.laser, self.slm, self._exposure = laser, slm, 5.0
        self.n, self.log = 0, []
    def sensor_shape(self): return (CAM_H, CAM_W)
    def dtype(self): return np.uint16
    def get_exposure(self): return self._exposure
    def set_exposure(self, exposure): self._exposure = exposure
    def snap(self, buffer):   # buffer = full sensor; the core crops it to the ROI
        self.n += 1
        buffer[:] = (self.n % 16) * 4096 + POSITION
        on_disk = bool(CURRENT_RUN and any((WORK / "data").glob(f"*/*_{CURRENT_RUN}_*/{CURRENT_RUN}.tif")))
        self.log.append((self.laser.get_property_value("S"), self.laser.get_property_value("P"),
                         self.slm.shown, on_disk))
        time.sleep(self._exposure / 1000.0)
        return {}


POSITION = np.arange(CAM_H * CAM_W, dtype=np.uint16).reshape(CAM_H, CAM_W)
CURRENT_RUN = None   # name of the run in progress, for the "TIFF on disk yet?" log
FULL = (0, 0, CAM_W, CAM_H)


def make_core():
    """A simulated microscope. pymmcore-plus picks the core's signal backend
    when it's created: before any Qt app (headless) -> plain signals, after
    one (as imaging_gui does) -> thread-safe Qt signals."""
    global mmc, slm_dev, laser, cam
    mmc = UniMMCore()
    slm_dev, laser = SimSLM(), SimLaser()
    cam = SimCamera(laser, slm_dev)
    mmc.loadPyDevice("Cam", cam); mmc.loadPyDevice("SLM", slm_dev); mmc.loadPyDevice("L", laser)
    mmc.initializeAllDevices(); mmc.setCameraDevice("Cam"); mmc.setSLMDevice("SLM")
    return mmc


make_core()   # headless parts: no Qt app yet, as in a real headless run

import experiments.timelapse as tl


def reset(run_name=None):
    global CURRENT_RUN
    cam.n, cam.log, CURRENT_RUN = 0, [], run_name


def run_folder(name):
    found = sorted((WORK / "data").glob(f"*/*_{name}_*"))
    return found[-1] if found else None


def check_run(folder, name, n_expected, n_requested, exposure, power, preset, label,
              roi=FULL, buffer="ram"):
    stack = tifffile.imread(folder / f"{name}.tif")
    meta = json.load(open(folder / f"{name}_metadata.json"))
    with tifffile.TiffFile(folder / f"{name}.tif") as f:
        axes = f.series[0].axes
    x, y, w, h = roi
    check(stack.shape == (n_expected, h, w) and axes == "TYX",
          f"{label}: {stack.shape} {axes} stack (ROI {w}x{h})")
    check(np.array_equal(stack[:, 0, 0] // 4096, np.arange(1, n_expected + 1) % 16),
          f"{label}: every frame, in order")
    check(np.array_equal(stack[0] % 4096, POSITION[y:y + h, x:x + w])
          and meta["camera_roi_xywh"] == list(roi),
          f"{label}: frames are the camera ROI {list(roi)} (position on the sensor checked)")
    on_disk = [entry[3] for entry in cam.log[:n_expected]]
    check(meta["buffer"] == buffer and (not any(on_disk) if buffer == "ram" else all(on_disk)),
          f"{label}: buffer '{buffer}' - TIFF on disk during the run: "
          f"{'never' if not any(on_disk) else 'yes'}")
    check(meta["frames_requested"] == n_requested and meta["frames_acquired"] == n_expected
          and meta["completed"] == (n_expected == n_requested)
          and meta["exposure_ms"] == exposure and meta["laser_power_mW"] == power
          and meta["interval_s"] == 0.0 and len(meta["frame_times_ms"]) == n_expected,
          f"{label}: metadata (frames, exposure, laser, interval, timestamps)")
    check(meta["phase_mask"]["preset"] == preset
          and (preset is None or (folder / "slm_mask.tif").exists()),
          f"{label}: phase mask '{preset}' recorded")
    lit = list(cam.log)
    check(all(e[0] == "open" and e[1] == power for e in lit[:n_expected]),
          f"{label}: laser on at {power} mW for every frame")
    check(laser.get_property_value("S") == "closed", f"{label}: laser off afterwards")
    return meta, lit


# ------------------------------------------------------------------ headless
print("headless, no preview")
reset("beads_test_1")
with mock.patch.object(tl, "load_config", lambda: config), \
     mock.patch.object(tl, "load_mmcore", lambda c: mmc), \
     mock.patch.object(tl, "PREVIEW", False), mock.patch.object(tl, "PHASE_MASK", "edof"), \
     mock.patch.object(sys, "argv", ["timelapse", "beads test #1"]):
    tl.main()
folder = run_folder("beads_test_1")
check(folder is not None and folder.name.endswith("_beads_test_1_1")
      and folder.parent.name == folder.name[:8], f"run folder YYYYMMDD/YYYYMMDD_<name>_N ({folder})")
meta, lit = check_run(folder, "beads_test_1", 30, 30, 5.0, 20.0, "edof", "headless")
edof = build_imaging_mask(config, "edof")[0]
check(all(np.array_equal(e[2], edof) for e in lit), "headless: EDOF mask on the SLM throughout")
rate = meta["measured_sampling_rate_hz"]
check(rate is not None and 100 < rate <= 210,
      f"headless: continuous, measured {rate:.1f} Hz (<= {meta['max_sampling_rate_hz']:.0f} Hz)")

print("headless, streamed to disk (BUFFER = 'disk')")
reset("disk_run")
with mock.patch.object(tl, "load_config", lambda: config), \
     mock.patch.object(tl, "load_mmcore", lambda c: mmc), \
     mock.patch.object(tl, "PREVIEW", False), mock.patch.object(tl, "PHASE_MASK", "flat"), \
     mock.patch.object(tl, "BUFFER", "disk"):
    tl.main("disk_run")
check_run(run_folder("disk_run"), "disk_run", 30, 30, 5.0, 20.0, "flat", "disk", buffer="disk")

print("headless refusals")
n_folders = len(list((WORK / "data").glob("*/*")))
for kwargs, why, exc in [({"name": "  "}, "empty name", ValueError)]:
    try:
        with mock.patch.object(tl, "load_config", lambda: config), \
             mock.patch.object(sys, "argv", ["timelapse"]):
            tl.main(**kwargs)
        check(False, f"{why} refused")
    except exc:
        check(True, f"{why} refused")
shutil.move(config.phase_masks_path, WORK / "moved.json")
try:
    with mock.patch.object(tl, "load_config", lambda: config), \
         mock.patch.object(tl, "load_mmcore", lambda c: (_ for _ in ()).throw(AssertionError)), \
         mock.patch.object(tl, "PHASE_MASK", "aberration_correction"):
        tl.main("x")
    check(False, "missing correction refused")
except ValueError:
    check(True, "missing stored correction refused before touching hardware")
shutil.move(WORK / "moved.json", config.phase_masks_path)
check(len(list((WORK / "data").glob("*/*"))) == n_folders, "...and no run folder created")
try:
    with mock.patch.object(tl, "load_config", lambda: config), \
         mock.patch.object(tl, "load_mmcore", lambda c: mmc), \
         mock.patch.object(tl, "PHASE_MASK", "flat"):
        tl.main("too_big", num_frames=10**9)
    check(False, "run too big for RAM refused")
except MemoryError as exc:
    check("GB of RAM" in str(exc), f"run too big for RAM refused ({str(exc)[:70]}...)")
check(len(list((WORK / "data").glob("*/*"))) == n_folders, "...before creating a run folder")

print("headless with preview, closed early")
import napari
from qtpy.QtWidgets import QApplication


def pump(seconds, until=None):
    app = QApplication.instance()
    t0 = time.time()
    while time.time() - t0 < seconds and not (until and until()):
        app.processEvents(); time.sleep(0.01)


def close_after_some_frames():
    pump(10, until=lambda: cam.n >= 40)
    viewer = napari.current_viewer()
    pump(10, until=lambda: len(viewer.layers) > 0)  # first refresh can be slow (software GL)
    shown_layers.append([layer.name for layer in viewer.layers])
    viewer.close()


reset("early_stop")
shown_layers = []
with mock.patch.object(tl, "load_config", lambda: config), \
     mock.patch.object(tl, "load_mmcore", lambda c: mmc), \
     mock.patch.object(tl, "PREVIEW", True), mock.patch.object(tl, "PHASE_MASK", "flat"), \
     mock.patch.object(napari, "run", close_after_some_frames):
    tl.main("early_stop", num_frames=2000)
folder = run_folder("early_stop")
meta = json.load(open(folder / "early_stop_metadata.json"))
n = meta["frames_acquired"]
check(shown_layers == [["Latest frame"]], f"preview showed the latest frame ({shown_layers})")
check(40 <= n < 2000 and not meta["completed"], f"stopped early after {n} frames")
check_run(folder, "early_stop", n, 2000, 5.0, 20.0, "flat", "early stop")

# ------------------------------------------------------------------ GUI
print("napari GUI 'Timelapse' panel")
from napari_micromanager._gui_objects._toolbar import DOCK_WIDGETS as TOOLBAR
from pymmcore_widgets.control._camera_roi_widget import CUSTOM_ROI, CameraRoiWidget
from qtpy.QtWidgets import QDoubleSpinBox, QLabel, QLineEdit, QPushButton, QSpinBox

GUI_ROI = (16, 8, 32, 24)   # x, y, w, h - off-centre, so a wrong offset would show

import experiments.imaging_gui as gui
import experiments.load_phase_mask as lpm

results = {}


def drive_gui():
    viewer = napari.current_viewer()
    pump(0.3)
    results["signals"] = type(mmc.events).__name__
    panel = viewer.window._dock_widgets["Timelapse"].widget()
    name = panel.findChild(QLineEdit)
    spins = panel.findChildren(QDoubleSpinBox)
    exposure = next(s for s in spins if s.suffix() == " ms")
    power = next(s for s in spins if s.suffix() == " mW")
    frames = panel.findChild(QSpinBox)
    buttons = {b.text(): b for b in panel.findChildren(QPushButton)}
    labels = panel.findChildren(QLabel)

    def label_text(fragment):
        return next((lab.text() for lab in labels if fragment in lab.text()), "")

    results["defaults"] = (exposure.value(), power.value(), frames.value())
    exposure.setValue(4.0)
    results["rate_label"] = label_text("Hz")
    exposure.setValue(5.0)

    # No name -> refused.
    buttons["Start timelapse"].click(); pump(0.2)
    results["no_name"] = label_text("name")

    # Crop the sensor with the Camera ROI widget napari-micromanager docks
    # under its crop icon, as the scientist would.
    roi_widget = CameraRoiWidget(mmcore=mmc)
    roi_widget.camera_roi_combo.setCurrentText(CUSTOM_ROI)
    roi_widget.roi_width.setValue(GUI_ROI[2]); roi_widget.roi_height.setValue(GUI_ROI[3])
    roi_widget.start_x.setValue(GUI_ROI[0]); roi_widget.start_y.setValue(GUI_ROI[1])
    roi_widget.crop_btn.click()
    pump(0.2)
    results["roi_after_crop"] = tuple(mmc.getROI())

    # Load the correction from the mask panel, then a full run.
    masks = viewer.window._dock_widgets["SLM phase mask"].widget()
    next(b for b in masks.findChildren(QPushButton) if b.text() == "Aberration correction").click()
    pump(0.2)
    name.setText("gui run"); power.setValue(35.0); frames.setValue(25)
    reset("gui_run")
    buttons["Start timelapse"].click()
    results["inputs_locked"] = not name.isEnabled() and buttons["Stop"].isEnabled()
    pump(20, until=lambda: "Done" in label_text("saved"))
    results["done"] = label_text("saved")
    results["log_full"] = list(cam.log)
    results["unlocked"] = name.isEnabled() and buttons["Start timelapse"].isEnabled()

    # A long run, stopped.
    name.setText("gui stop"); frames.setValue(5000)
    reset("gui_stop")
    buttons["Start timelapse"].click()
    pump(10, until=lambda: cam.n >= 30)
    buttons["Stop"].click()
    pump(20, until=lambda: "Stopped" in label_text("saved"))
    results["stopped"] = label_text("saved")
    results["layers"] = [layer.name for layer in viewer.layers]
    viewer.close()


with mock.patch.object(gui, "load_config", lambda: config), \
     mock.patch.object(gui, "load_mmcore", lambda c: make_core()), \
     mock.patch.object(gui.napari, "run", drive_gui), \
     mock.patch.object(lpm, "load_config", lambda: config), \
     mock.patch.object(tl, "load_config", lambda: config), \
     mock.patch.object(tl, "load_mmcore", lambda c: (_ for _ in ()).throw(AssertionError)):
    gui.main()

check(TOOLBAR["Camera ROI"][0] is CameraRoiWidget,
      "napari-micromanager's 'Camera ROI' (crop) tool is the CameraRoiWidget used here")
check(results["roi_after_crop"] == GUI_ROI, f"crop applied to the core: {results['roi_after_crop']}")
check(results["signals"] == "QCoreSignaler", f"GUI core has Qt signals ({results['signals']})")
check(results["defaults"] == (5.0, 20.0, 30), f"panel defaults from the config {results['defaults']}")
check("250.00 Hz" in results["rate_label"], f"sampling rate shown: {results['rate_label']!r}")
check("name" in results["no_name"].lower(), "Start without a name refused")
check(results["inputs_locked"] and results["unlocked"], "form locked while running, unlocked after")
check("Done: 25/25 frames saved" in results["done"] and "Hz" in results["done"],
      f"panel reports: {results['done']!r}")
cam.log = results["log_full"]
folder = run_folder("gui_run")
check_run(folder, "gui_run", 25, 25, 5.0, 35.0, "aberration_correction", "GUI", roi=GUI_ROI)
ao = build_imaging_mask(config, "aberration_correction")[0]
check(all(np.array_equal(e[2], ao) for e in results["log_full"][:25]),
      "GUI: run with the mask loaded from the 'SLM phase mask' panel")
check(results["stopped"].startswith("Stopped:"), f"Stop: {results['stopped']!r}")
meta = json.load(open(run_folder("gui_stop") / "gui_stop_metadata.json"))
stack = tifffile.imread(run_folder("gui_stop") / "gui_stop.tif")
check(not meta["completed"] and stack.shape[0] == meta["frames_acquired"] >= 30,
      f"Stop keeps the {meta['frames_acquired']} frames so far")
check(laser.get_property_value("S") == "closed", "laser off after the GUI closes")
print(f"  (viewer layers: {results['layers']})")

print()
assert not failures, failures
print("TIMELAPSE TEST PASSED")
