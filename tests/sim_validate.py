"""
Validation: simulate a pupil at a KNOWN centre, run the real
phase-mask / localization / map-analysis code against it, and check
the recovered centre.
"""
import json
import sys
import numpy as np

sys.path.insert(0, "/home/claude/reflex3d_controller")

from core.config import MicroscopeConfig
from functions.phase_masks import zernike_patch_mask, raster_positions
from functions.localization import localize_psf, squared_separation
from calibration.map_analysis import estimate_fourier_center

# --- simulated instrument ---
N_SLM = 512          # simulated SLM is 512x512
N_FFT = 1024         # zero-padded FFT grid -> well-sampled PSF
R_PUPIL = 60
TRUE_CX, TRUE_CY = 214.0, 166.0   # ground truth, deliberately off-centre

cfg_dict = {
    "slm": {"resolution": [N_SLM, N_SLM], "flat_value": 0,
            "grey_level_2pi": 255},
    "fourier_plane": {"radius_px": R_PUPIL},
}
cfg_path = "/tmp/sim_config.json"
with open(cfg_path, "w") as f:
    json.dump(cfg_dict, f)
config = MicroscopeConfig(cfg_path)

yy, xx = np.mgrid[0:N_SLM, 0:N_SLM]
PUPIL = ((xx - TRUE_CX) ** 2 + (yy - TRUE_CY) ** 2) <= R_PUPIL ** 2


def simulate_frame(mask_grey, noise_e=2.0, photons=2.0e6, rng=None):
    """Grey-level SLM mask -> simulated camera frame via FFT."""
    rng = rng or np.random.default_rng(0)
    phase = mask_grey.astype(float) / 255.0 * 2.0 * np.pi
    field = np.zeros((N_FFT, N_FFT), dtype=complex)
    field[:N_SLM, :N_SLM] = PUPIL * np.exp(1j * phase)
    psf = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2
    psf = psf / psf.sum() * photons
    # crop a camera-like ROI around the nominal focus
    c = N_FFT // 2
    half = 96
    roi = psf[c - half:c + half, c - half:c + half]
    return roi + rng.normal(0.0, noise_e, roi.shape)


def build_map(mode, amplitude_rad, patch_d, step, loc_method):
    xs, ys = raster_positions(config, step)
    d2 = np.full((len(ys), len(xs)), np.nan)
    rng = np.random.default_rng(42)
    for iy, ycen in enumerate(ys):
        for ix, xcen in enumerate(xs):
            locs = []
            for sign in (+1, -1):
                m = zernike_patch_mask(config, (xcen, ycen), patch_d,
                                       mode, amplitude_rad, sign)
                frame = simulate_frame(m, rng=rng)
                locs.append(localize_psf(frame, method=loc_method,
                                         min_snr=2.0))
            d2[iy, ix] = squared_separation(locs[0], locs[1])
    return xs, ys, d2


def report(tag, mode, xs, ys, d2, loc_method):
    res = estimate_fourier_center(d2, xs, ys, mode)
    ex, ey = res["center_x"], res["center_y"]
    err = np.hypot(ex - TRUE_CX, ey - TRUE_CY)
    finite = d2[np.isfinite(d2)]
    print(f"\n=== {tag} (mode={mode}, loc={loc_method}) ===")
    print(f"  d2 range: {finite.min():.3f} .. {finite.max():.3f}")
    print(f"  support pts: {res['n_support_points']}, "
          f"failed: {res['n_failed_points']}, method: {res['method']}")
    print(f"  estimate: ({ex:.1f}, {ey:.1f})  truth: "
          f"({TRUE_CX}, {TRUE_CY})  ERROR: {err:.1f} px")
    for w in res["warnings"]:
        print(f"  WARN: {w}")
    return err


if __name__ == "__main__":
    PATCH = 60
    errs = {}
    for mode, amp, STEP in [("tilt_x", 2 * np.pi, 24),
                            ("defocus", 1.0 * np.pi, 20)]:
        for loc in ["frame_centroid", "windowed"]:
            xs, ys, d2 = build_map(mode, amp, PATCH, STEP, loc)
            errs[(mode, loc, STEP)] = report(f"{mode}/{loc}", mode, xs, ys, d2, loc)
    print("\n--- summary (error in SLM px) ---")
    for k, v in errs.items():
        print(f"  {k}: {v:.1f}")
