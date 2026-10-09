"""
End-to-end test of the Fourier-plane refinement path (windowed fine
raster -> radial-symmetry centre -> config write-back) on the simulated
microscope of tests/e2e_test.py, starting from a deliberately offset
"coarse" centre.

    python tests/refinement_e2e_test.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e_test as sim  # simulated instrument + config (module-level setup)

from core.config import MicroscopeConfig
from core.file_io import save_alignment_map, save_figure, write_fourier_center
from core.logging_setup import setup_logger
from calibration.fourier_alignment import acquire_d2_map, verify_point_source
from calibration.fourier_refinement import calibrated_channels, stored_center
from calibration.map_analysis import estimate_fourier_center, refine_fourier_center
from calibration.map_preview import build_refinement_figure
from functions.mmcore_camera import set_channel_roi
from functions.mmcore_slm import MMCoreSLM
from functions.phase_masks import local_raster_positions

COARSE_OFFSET = (13.0, -11.0)   # how wrong the "coarse" centre is (px)
PATCH, STEP = 60, 6             # fine step (coarse would be 24)


def run(mode, amplitude, channel="left"):
    config = MicroscopeConfig(sim.cfg_path)
    start = (sim.TRUE_CX + COARSE_OFFSET[0], sim.TRUE_CY + COARSE_OFFSET[1])
    write_fourier_center(config.path, channel, start,
                         extra={"calibration_mode": mode})
    config = MicroscopeConfig(sim.cfg_path)
    assert calibrated_channels(config) == [channel]   # 'right' is still null
    previous = stored_center(config, channel)
    assert previous == start

    mmc = sim.build_core()
    run_folder = sim.WORK / f"refine_{mode}"; run_folder.mkdir(exist_ok=True)
    logger = setup_logger(run_folder)

    window_r = config.get("fourier_plane", "radius_px")
    xs, ys, window = local_raster_positions(config, previous, window_r, STEP)
    n_pos = int(window.sum())
    assert 0.9 * np.pi * (window_r / STEP) ** 2 < n_pos < 1.1 * np.pi * (window_r / STEP) ** 2
    assert previous[0] in xs and previous[1] in ys   # old centre is a node

    with MMCoreSLM(mmc, config) as slm:
        set_channel_roi(mmc, config, channel)
        verify_point_source(mmc, config, slm, logger, 1, 1.0, "frame_centroid")
        xs, ys, d2, meta = acquire_d2_map(
            mmc, config, slm, logger, mode=mode, amplitude_rad=amplitude,
            patch_diameter_px=PATCH, step_px=STEP, exposure_ms=1,
            laser_power_mW=1.0, localization_method="frame_centroid",
            positions=(xs, ys), position_mask=window, use_tqdm=True,
        )
    assert meta["n_positions"] == n_pos
    assert int(np.isfinite(d2).sum()) + meta["n_failed_localizations"] == n_pos
    assert np.all(np.isnan(d2[~window])), "positions outside the window were acquired"
    assert mmc.getProperty("L", "S") == "closed"
    save_alignment_map(run_folder, xs, ys, d2, meta)

    res = refine_fourier_center(d2, xs, ys, mode, previous, window_r)
    fig = build_refinement_figure(xs, ys, d2, res, config, channel)
    save_figure(run_folder, fig, filename="fourier_refinement_preview.png")

    truth = np.array([sim.TRUE_CX, sim.TRUE_CY])
    err_before = np.hypot(*(np.array(previous) - truth))
    err_after = np.hypot(res["center_x"] - truth[0], res["center_y"] - truth[1])

    # For reference: what the whole-SLM estimator makes of the same window.
    ref = estimate_fourier_center(d2, xs, ys, mode, support_fraction=0.15)
    err_ref = np.hypot(ref["center_x"] - truth[0], ref["center_y"] - truth[1])
    print(f"\n[{mode}] {n_pos} positions; previous err {err_before:.1f} px -> "
          f"refined err {err_after:.1f} px (coarse estimator on same window: "
          f"{err_ref:.1f} px); explained {100 * res['explained_fraction']:.0f}%")
    for w in res["warnings"]:
        print(f"  WARN: {w}")
    return res, err_before, err_after


if __name__ == "__main__":
    res_t, b_t, a_t = run("tilt_x", 2 * np.pi)
    res_d, b_d, a_d = run("defocus", np.pi)

    cfg = MicroscopeConfig(sim.cfg_path)
    p, bak = write_fourier_center(cfg.path, "left",
                                  (res_t["center_x"], res_t["center_y"]),
                                  extra={"calibration_mode": "tilt_x"})
    after = json.load(open(p))
    assert after["fourier_plane"]["channels"]["right"]["center_px"] == [None, None]
    assert abs(after["fourier_plane"]["channels"]["left"]["center_px"][0]
               - res_t["center_x"]) < 1e-9
    print(f"\nERRORS: tilt {b_t:.1f} -> {a_t:.1f} px, "
          f"defocus {b_d:.1f} -> {a_d:.1f} px")
    assert a_t < 2.0 and a_t < b_t, "tilt refinement did not improve the centre"
    assert a_d < 6.0 and a_d < b_d, "defocus refinement did not improve the centre"
    print("REFINEMENT E2E PASSED")
