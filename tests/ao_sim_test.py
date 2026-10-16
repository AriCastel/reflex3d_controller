"""
Aberration correction against a simulated microscope with a KNOWN
pupil aberration: pymmcore-plus Python devices on a UniMMCore (camera
imaging the pupil + aberration + displayed SLM phase, SLM, laser),
with photon/read noise and photobleaching. Everything above the
device level is the real code. Checks that every power-aware metric
improves the PSF without losing power (second_moment, which is
power-blind, is only reported), and that the default metric
recovers the aberration (correction ~ -aberration).

    python tests/ao_sim_test.py
"""
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
from pymmcore_plus.experimental.unicore import (
    GenericDevice, SimpleCameraDevice, SLMDevice, UniMMCore,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import MicroscopeConfig
from core.file_io import save_ao_result, save_figure
from core.logging_setup import setup_logger
from calibration.fourier_alignment import verify_point_source
from functions.adaptive_optics import (
    ZernikePupil, acquire_dark_frame, build_ao_figure, coefficient_table,
    run_aberration_correction, set_guide_star_roi,
)
from functions.mmcore_camera import set_channel_roi
from functions.mmcore_laser import laser_on_mmcore, set_laser_power_mmcore
from functions.mmcore_slm import MMCoreSLM
from functions.psf_metrics import METRICS
from functions.zernike import noll_zernike

N_SLM, N_FFT, R_PUPIL = 256, 512, 50
CENTER = (120.0, 130.0)
SENSOR_H, SENSOR_W = 300, 400
ROI = (40, 60, 128)                    # left channel ROI x, y, size
BEAD_OFFSET = (9, -7)                  # bead position relative to ROI centre
ABERRATION = {5: 0.6, 7: -0.5, 8: 0.3, 11: 0.4}   # Noll j -> rad RMS
NOLL = [5, 6, 7, 8, 9, 10, 11]
PHOTONS, READ_NOISE, OFFSET, BLEACH_PER_SNAP = 4e5, 2.0, 100.0, 1.5e-4

WORK = Path("/tmp/ao_sim"); shutil.rmtree(WORK, ignore_errors=True); WORK.mkdir(parents=True)
CFG = WORK / "config.json"
json.dump({
    "camera": {"channel_rois": {"left": {"position_px": list(ROI[:2]), "size_px": ROI[2]}}},
    "slm": {"device_label": "SLM", "resolution": [N_SLM, N_SLM], "flat_value": 0,
            "grey_level_2pi": 255, "refresh_rate_hz": 30, "settle_frames": 0},
    "laser": {"device_label": "L", "power_property": "P", "state_property": "S"},
    "fourier_plane": {"radius_px": R_PUPIL,
                      "channels": {"left": {"center_px": list(CENTER)}}},
}, open(CFG, "w"))
config = MicroscopeConfig(CFG)

yy, xx = np.mgrid[0:N_SLM, 0:N_SLM]
_rho = np.hypot(xx - CENTER[0], yy - CENTER[1]) / R_PUPIL
_theta = np.arctan2(yy - CENTER[1], xx - CENTER[0])
PUPIL = _rho <= 1
ABERRATION_PHASE = sum(c * noll_zernike(j, _rho, _theta) for j, c in ABERRATION.items())


class SimSLM(SLMDevice):
    def __init__(self):
        super().__init__(); self.loaded = self.shown = np.zeros((N_SLM, N_SLM), np.uint8)
    def shape(self): return (N_SLM, N_SLM)
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
    def __init__(self, slm, laser, rng):
        super().__init__(); self.slm, self.laser, self.rng = slm, laser, rng
        self._exposure, self.lit_snaps = 20.0, 0
    def sensor_shape(self): return (SENSOR_H, SENSOR_W)
    def dtype(self): return np.uint16
    def get_exposure(self): return self._exposure
    def set_exposure(self, exposure): self._exposure = exposure
    def snap(self, buffer):
        signal = np.zeros(buffer.shape)
        if self.laser.get_property_value("S") == "open":
            phase = self.slm.shown.astype(float) / 255.0 * 2 * np.pi + ABERRATION_PHASE
            field = np.zeros((N_FFT, N_FFT), complex)
            field[:N_SLM, :N_SLM] = PUPIL * np.exp(1j * phase)
            psf = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2
            bleach = np.exp(-BLEACH_PER_SNAP * self.lit_snaps)
            self.lit_snaps += 1
            psf *= PHOTONS * bleach / (np.abs(PUPIL).sum() * N_FFT ** 2)
            c, h = N_FFT // 2, ROI[2] // 2
            x0 = ROI[0] - BEAD_OFFSET[0]; y0 = ROI[1] - BEAD_OFFSET[1]
            signal[y0:y0 + ROI[2], x0:x0 + ROI[2]] = psf[c - h:c + h, c - h:c + h]
        noisy = self.rng.poisson(signal) + self.rng.normal(OFFSET, READ_NOISE, buffer.shape)
        buffer[:] = np.clip(noisy, 0, 65535)
        return {}


def run(metric, iterations=100, seed=1):
    rng = np.random.default_rng(seed)
    mmc = UniMMCore()
    slm_dev, laser = SimSLM(), SimLaser()
    mmc.loadPyDevice("Cam", SimCamera(slm_dev, laser, rng))
    mmc.loadPyDevice("SLM", slm_dev); mmc.loadPyDevice("L", laser)
    mmc.initializeAllDevices(); mmc.setCameraDevice("Cam"); mmc.setSLMDevice("SLM")

    folder = WORK / type(metric).__name__; folder.mkdir()
    logger = setup_logger(folder)
    with MMCoreSLM(mmc, config) as slm:
        roi = set_channel_roi(mmc, config, "left")
        frame, loc = verify_point_source(mmc, config, slm, logger, 20, 1.0, "windowed")
        gs = set_guide_star_roi(mmc, (roi[0] + loc["x"], roi[1] + loc["y"]), 48)
        dark = acquire_dark_frame(mmc, 10)
        set_laser_power_mmcore(mmc, config, 1.0); laser_on_mmcore(mmc, config)
        pupil = ZernikePupil(config, "left", NOLL)
        result = run_aberration_correction(
            mmc, config, slm, pupil, metric, dark, logger, settle_s=0,
            iterations=iterations, gain=0.1, perturbation=0.3, patience=25,
            max_abs=3.0, seed=seed, progress_every=50)
    save_ao_result(folder, result, {"metric": type(metric).__name__})
    save_figure(folder, build_ao_figure(result, NOLL, type(metric).__name__, "left"),
                "ao_correction.png")
    return result, gs


if __name__ == "__main__":
    truth = np.array([-ABERRATION.get(j, 0.0) for j in NOLL])
    failures = []
    for name, cls in METRICS.items():
        result, gs = run(cls())
        err = np.sqrt(np.mean((result["coefficients"] - truth) ** 2))
        print(f"\n=== {name} === guide-star ROI {gs}")
        print(f"  metric {result['metric_original']:.4g} -> {result['metric_corrected']:.4g}; "
              f"power ratio {result['power_ratio']:.2f}; RMS coeff error {err:.3f} rad; "
              f"{result['n_snaps']} snaps in {result['elapsed_s']:.1f} s")
        if name == "power_weighted_second_moment":
            print(coefficient_table(NOLL, result["coefficients"]))
            print("  truth (-aberration):", np.round(truth, 3))
            if err > 0.15:
                failures.append(f"{name}: coefficient error {err:.3f} rad")
        if name == "second_moment":
            # Power-blind: expected to wander off to a dimmer PSF. Reported
            # for comparison only.
            continue
        if not result["metric_corrected"] < result["metric_original"]:
            failures.append(f"{name}: metric did not improve")
        if result["power_ratio"] < 0.9:
            failures.append(f"{name}: lost power ({result['power_ratio']:.2f})")
    print(f"\nOutputs in {WORK}")
    assert not failures, failures
    print("AO SIM PASSED")
