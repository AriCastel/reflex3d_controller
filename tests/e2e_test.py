"""
End-to-end test of the real acquisition -> analysis -> save -> config
write-back path, against a simulated microscope. The hardware is
replaced by pymmcore-plus Python devices on a UniMMCore: a camera
whose image is the PSF produced by the pupil + whatever mask the SLM
is displaying, an SLM, and a laser. Everything above the device level
(MMCoreSLM, channel ROIs, mmcore_laser, acquire_d2_map, ...) is the
real code.

    python tests/e2e_test.py
"""
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
from pymmcore_plus.experimental.unicore import (
    GenericDevice,
    SimpleCameraDevice,
    SLMDevice,
    UniMMCore,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import MicroscopeConfig
from core.file_io import save_alignment_map, save_figure, write_fourier_center
from core.logging_setup import setup_logger
from calibration.fourier_alignment import acquire_d2_map, verify_point_source
from calibration.map_analysis import consistency_check, estimate_fourier_center
from calibration.map_preview import build_map_figure
from functions.mmcore_camera import set_channel_roi
from functions.mmcore_slm import MMCoreSLM

# ---------- simulated instrument ----------
N_SLM, N_FFT, R_PUPIL = 512, 1024, 60
TRUE_CX, TRUE_CY = 214.0, 166.0
SENSOR_H, SENSOR_W = 600, 900
ROI_HALF = 128
LEFT_ROI = (100, 150)    # top-left (x, y) of the left channel's ROI
RIGHT_ROI = (550, 150)

WORK = Path("/tmp/e2e"); shutil.rmtree(WORK, ignore_errors=True)
WORK.mkdir(parents=True)
cfg_path = WORK / "sim_config.json"
json.dump({
    "camera": {"channel_rois": {
        "left": {"position_px": list(LEFT_ROI), "size_px": 2 * ROI_HALF},
        "right": {"position_px": list(RIGHT_ROI), "size_px": 2 * ROI_HALF},
    }},
    "slm": {"device_label": "SLM", "resolution": [N_SLM, N_SLM],
            "flat_value": 0, "grey_level_2pi": 255,
            "refresh_rate_hz": 30, "settle_frames": 0},
    "laser": {"device_label": "L", "power_property": "P", "state_property": "S"},
    "fourier_plane": {"radius_px": R_PUPIL,
                      "channels": {"left": {"center_px": [None, None]},
                                   "right": {"center_px": [None, None]}}},
    "fourier_alignment": {"support_fraction": 0.15},
}, open(cfg_path, "w"), indent=2)
config = MicroscopeConfig(cfg_path)

yy, xx = np.mgrid[0:N_SLM, 0:N_SLM]
PUPIL = ((xx - TRUE_CX) ** 2 + (yy - TRUE_CY) ** 2) <= R_PUPIL ** 2
RNG = np.random.default_rng(7)


class SimSLM(SLMDevice):
    def __init__(self):
        super().__init__()
        self.loaded = self.shown = np.zeros((N_SLM, N_SLM), np.uint8)
    def shape(self): return (N_SLM, N_SLM)
    def dtype(self): return np.uint8
    def set_image(self, pixels): self.loaded = np.array(pixels)
    def display_image(self): self.shown = self.loaded
    def set_exposure(self, interval_ms): pass
    def get_exposure(self): return 0.0


class SimCamera(SimpleCameraDevice):
    """PSF of the pupil under the displayed SLM mask, in the left ROI."""
    def __init__(self, slm):
        super().__init__(); self.slm = slm; self._exposure = 100.0
    def sensor_shape(self): return (SENSOR_H, SENSOR_W)
    def dtype(self): return np.float64
    def get_exposure(self): return self._exposure
    def set_exposure(self, exposure): self._exposure = exposure
    def snap(self, buffer):
        phase = self.slm.shown.astype(float) / 255.0 * 2 * np.pi
        f = np.zeros((N_FFT, N_FFT), complex)
        f[:N_SLM, :N_SLM] = PUPIL * np.exp(1j * phase)
        psf = np.abs(np.fft.fftshift(np.fft.fft2(f))) ** 2
        psf = psf / psf.sum() * 2e6
        c = N_FFT // 2
        buffer[:] = RNG.normal(0, 2.0, buffer.shape)
        x0, y0 = LEFT_ROI
        buffer[y0:y0 + 2 * ROI_HALF, x0:x0 + 2 * ROI_HALF] += \
            psf[c - ROI_HALF:c + ROI_HALF, c - ROI_HALF:c + ROI_HALF]
        return {}


class SimLaser(GenericDevice):
    def __init__(self):
        super().__init__()
        self.register_property("P", default_value=0.0)
        self.register_property("S", default_value="closed",
                               allowed_values=["open", "closed"])


def build_core():
    mmc = UniMMCore()
    slm = SimSLM()
    mmc.loadPyDevice("Cam", SimCamera(slm))
    mmc.loadPyDevice("SLM", slm)
    mmc.loadPyDevice("L", SimLaser())
    mmc.initializeAllDevices()
    mmc.setCameraDevice("Cam")
    mmc.setSLMDevice("SLM")
    return mmc


def run(mode, amplitude, step, patch, channel="left"):
    mmc = build_core()
    run_folder = WORK / f"run_{mode}"; run_folder.mkdir(exist_ok=True)
    logger = setup_logger(run_folder)

    with MMCoreSLM(mmc, config) as slm:
        roi = set_channel_roi(mmc, config, channel)
        frame, loc = verify_point_source(mmc, config, slm, logger, 1, 1.0,
                                         "frame_centroid")
        assert frame.shape == (2 * ROI_HALF, 2 * ROI_HALF), frame.shape
        assert loc is not None, "bead not localized inside the ROI"
        xs, ys, d2, meta = acquire_d2_map(
            mmc, config, slm, logger, mode=mode, amplitude_rad=amplitude,
            patch_diameter_px=patch, step_px=step, exposure_ms=1,
            laser_power_mW=1.0, localization_method="frame_centroid",
            progress_every=0,
        )
    assert meta["camera_roi_xywh"] == list(roi)
    meta["channel"] = channel
    save_alignment_map(run_folder, xs, ys, d2, meta)

    res = estimate_fourier_center(d2, xs, ys, mode, support_fraction=0.15)
    res["patch_diameter_px"] = patch
    res = consistency_check(res, config)

    fig = build_map_figure(xs, ys, d2, res, config, channel)
    save_figure(run_folder, fig)

    cx, cy = res["center_x"], res["center_y"]
    err = np.hypot(cx - TRUE_CX, cy - TRUE_CY)
    laser_state = mmc.getProperty("L", "S")
    print(f"\n[{mode}] estimate=({cx:.1f},{cy:.1f}) truth=({TRUE_CX},{TRUE_CY}) "
          f"err={err:.1f}px method={res['method']}")
    print(f"  camera ROI: {roi}; laser left {laser_state}")
    print(f"  files: {sorted(p.name for p in run_folder.iterdir())}")
    for w in res["warnings"]:
        print(f"  WARN: {w}")
    assert laser_state == "closed", "laser left on after the raster"
    return res, err


if __name__ == "__main__":
    res_t, err_t = run("tilt_x", 2 * np.pi, 24, 60)
    res_d, err_d = run("defocus", np.pi, 20, 60)

    # config write-back
    cx, cy = res_t["center_x"], res_t["center_y"]
    p, bak = write_fourier_center(config.path, "left", (cx, cy),
                                  extra={"calibration_mode": "tilt_x"})
    after = json.load(open(p))
    print("\n--- config write-back ---")
    print("  left  :", after["fourier_plane"]["channels"]["left"])
    print("  right :", after["fourier_plane"]["channels"]["right"])
    print("  other keys preserved:", sorted(after.keys()))
    print("  backup exists:", Path(bak).exists())
    assert after["fourier_plane"]["channels"]["right"]["center_px"] == [None, None]
    assert after["fourier_plane"]["radius_px"] == R_PUPIL
    assert "fourier_alignment" in after
    assert after["camera"]["channel_rois"]["left"]["size_px"] == 2 * ROI_HALF
    print("\nERRORS: tilt %.1f px, defocus %.1f px" % (err_t, err_d))
    assert err_t < 8 and err_d < 15, "center recovery worse than expected"
    print("E2E PASSED")
