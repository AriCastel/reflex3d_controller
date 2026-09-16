"""
End-to-end test of the real acquisition -> analysis -> save -> config
write-back path, against a simulated microscope. Replaces only the two
hardware boundaries: pycromanager Core, and the SLM display window.
"""
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, "/home/claude/reflex3d_controller")

from core.config import MicroscopeConfig
from core.file_io import save_alignment_map, save_figure, write_fourier_center
from core.logging_setup import setup_logger
from calibration.fourier_alignment import acquire_d2_map, estimate_duration_s, format_duration
from calibration.map_analysis import consistency_check, estimate_fourier_center
from calibration.map_preview import build_map_figure

# ---------- simulated instrument ----------
N_SLM, N_FFT, R_PUPIL = 512, 1024, 60
TRUE_CX, TRUE_CY = 214.0, 166.0

WORK = Path("/tmp/e2e"); shutil.rmtree(WORK, ignore_errors=True)
WORK.mkdir(parents=True)
cfg_path = WORK / "sim_config.json"
json.dump({
    "slm": {"resolution": [N_SLM, N_SLM], "flat_value": 0,
            "grey_level_2pi": 255, "refresh_rate_hz": 30, "settle_frames": 0},
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


class MockSLM:
    def __init__(self): self.current = np.zeros((N_SLM, N_SLM), np.uint8)
    def show_mask(self, m): self.current = m


class MockCore:
    """Simulates snap_image() by FFT-propagating the current SLM mask."""
    def __init__(self, slm): self.slm = slm; self.exposure = 100; self.props = {}
    def set_exposure(self, e): self.exposure = e
    def set_property(self, d, p, v): self.props[(d, p)] = v
    def snap_image(self):
        phase = self.slm.current.astype(float) / 255.0 * 2 * np.pi
        f = np.zeros((N_FFT, N_FFT), complex)
        f[:N_SLM, :N_SLM] = PUPIL * np.exp(1j * phase)
        psf = np.abs(np.fft.fftshift(np.fft.fft2(f))) ** 2
        psf = psf / psf.sum() * 2e6
        c, half = N_FFT // 2, 128
        self._img = (psf[c - half:c + half, c - half:c + half]
                     + RNG.normal(0, 2.0, (2 * half, 2 * half)))
    def get_tagged_image(self):
        class TI: pass
        ti = TI(); ti.pix = self._img.ravel()
        ti.tags = {"Width": self._img.shape[1], "Height": self._img.shape[0]}
        return ti


def run(mode, amplitude, step, patch, channel="left"):
    slm = MockSLM(); core = MockCore(slm)
    run_folder = WORK / f"run_{mode}"; run_folder.mkdir(exist_ok=True)
    logger = setup_logger(run_folder)

    xs, ys, d2, meta = acquire_d2_map(
        core, config, slm, logger, mode=mode, amplitude_rad=amplitude,
        patch_diameter_px=patch, step_px=step, exposure_ms=1,
        laser_power_mW=1.0, localization_method="frame_centroid",
        progress_every=0,
    )
    meta["channel"] = channel
    map_path = save_alignment_map(run_folder, xs, ys, d2, meta)

    res = estimate_fourier_center(d2, xs, ys, mode,
                                  support_fraction=0.15)
    res["patch_diameter_px"] = patch
    res = consistency_check(res, config)

    fig = build_map_figure(xs, ys, d2, res, config, channel)
    png = save_figure(run_folder, fig)

    cx, cy = res["center_x"], res["center_y"]
    err = np.hypot(cx - TRUE_CX, cy - TRUE_CY)
    print(f"\n[{mode}] estimate=({cx:.1f},{cy:.1f}) truth=({TRUE_CX},{TRUE_CY}) "
          f"err={err:.1f}px method={res['method']}")
    print(f"  laser left off: {core.props.get(('L','S'))==0}")
    print(f"  files: {sorted(p.name for p in run_folder.iterdir())}")
    for w in res["warnings"]:
        print(f"  WARN: {w}")
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
    print("\nERRORS: tilt %.1f px, defocus %.1f px" % (err_t, err_d))
    assert err_t < 8 and err_d < 15, "center recovery worse than expected"
    print("E2E PASSED")
