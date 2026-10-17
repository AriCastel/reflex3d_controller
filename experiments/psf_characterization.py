"""
experiments/psf_characterization.py

3D PSF characterization: a depth (Z) stack of a guide star (an isolated
microsphere), taken by stepping the micromanipulator the sample is
mounted on along the camera's depth axis, with a chosen phase mask on
the SLM (functions/imaging_masks.py presets; flat by default).

Acknowledgement: Luis Bastidas was the first to work out how to drive
our Sutter MPC micromanipulator and wrote the original Z-stack script
(zDepthVariation.py, SPIM 2) that this experiment is based on.

Flow
----
1. Live full-frame preview (flat mask) with the channel ROIs boxed -
   position the guide star, then close the window.
2. Choose the channel. The camera is cropped to its ROI, the guide star
   is localized and shown for confirmation, then the camera is cropped
   to a square around it (psf_characterization.guide_star_roi_px).
3. Choose the phase mask (Enter = flat); it's shown on both channels.
4. Confirm the scan (range, step, planes, estimated time). The stage
   overshoots ABOVE the top of the range by a few steps, comes back
   down to it, then steps DOWN through the range - one direction only,
   because of backlash - snapping a frame at each plane, laser on.
5. Laser off, same ROI: `dark_frames` frames are averaged and that mean
   is subtracted from every plane. The stage goes back to its starting
   depth (approached from above, like the scan).
6. Saved: psf_zstack.tif (float32, dark-subtracted, ZYX, top of the
   range first) with psf_zstack_metadata.json (per-plane commanded and
   read-back depth, settings, mask), the dark mean, and the mask.

The depth axis is whichever stage property micromanipulator.axis_properties
maps 'depth' to - check it if the stage has been re-mounted.

Run with:
    python -m experiments.psf_characterization
"""
import time
from datetime import datetime

import numpy as np

from core.config import load_config
from core.file_io import save_frame_tiff, save_phase_mask, save_zstack_tiff
from core.logging_setup import setup_logger
from core.mmcore import load_mmcore
from core.session import Session
from calibration.fourier_alignment import (
    format_duration,
    preview_sample_positioning,
    prompt_choice,
    prompt_yes_no,
    slm_settle_seconds,
    verify_point_source,
)
from functions.adaptive_optics import acquire_dark_frame, set_guide_star_roi
from functions.imaging_masks import PRESETS, build_imaging_mask, describe_imaging_mask
from functions.mmcore_camera import set_channel_roi
from functions.mmcore_laser import laser_off_mmcore, laser_on_mmcore, set_laser_power_mmcore
from functions.mmcore_micromanipulator import Micromanipulator, approach, scan, scan_positions
from functions.mmcore_slm import MMCoreSLM
from functions.napari_preview import show_verification_frame

# ---- experiment parameters (None -> use psf_characterization.* from config) ----
HALF_RANGE_UM = None
STEP_UM = None
EXPOSURE_MS = None
LASER_POWER_MW = None
DEFAULT_PHASE_MASK = "flat"


def main():
    config = load_config()
    pc = config["psf_characterization"]
    half_range = HALF_RANGE_UM if HALF_RANGE_UM is not None else pc["half_range_um"]
    step = STEP_UM if STEP_UM is not None else pc["step_um"]
    exposure_ms = EXPOSURE_MS or pc["exposure_ms"]
    laser_power = LASER_POWER_MW or pc["laser_power_mW"]

    session = Session(config, script_name="psf_characterization")
    logger = setup_logger(session.path)
    logger.info(f"Run folder: {session.path}")

    mmc = load_mmcore(config)
    logger.info("Connected to Micro-Manager via pymmcore-plus.")
    stage = Micromanipulator(mmc, config)
    stage.check_step(step)   # before touching anything
    logger.info(f"Depth axis = stage property '{stage.property_for('depth')}' "
                f"(micromanipulator.axis_properties)")

    with MMCoreSLM(mmc, config) as slm:
        moved = {}  # set once the stage starts moving, so it's returned even on an error
        try:
            out = _acquire(mmc, config, pc, slm, stage, logger, half_range, step,
                           exposure_ms, laser_power, moved)
        finally:
            laser_off_mmcore(mmc, config)
            start_depth = moved.get("start_depth_um")
            if start_depth is not None:
                approach(stage, "depth", start_depth, pc["overshoot_steps"] * step)
                logger.info(f"Stage back at its starting depth, {start_depth:.4f} um.")
            mmc.clearROI()
    if out is None:
        return

    stack = out["frames"] - out["dark"][None]
    offsets = out["commanded_um"] - out["start_depth_um"]
    metadata = {
        "finished_at": out["finished_at"],
        "channel": out["channel"],
        "phase_mask": out["mask_info"],
        "depth_axis_property": stage.property_for("depth"),
        "stage_axis_properties": stage.properties,
        "start_position_um": out["start_positions"],
        "half_range_um": half_range,
        "step_um": step,
        "overshoot_um": pc["overshoot_steps"] * step,
        "scan_direction": "descending (first plane = top of the range)",
        "z_offset_um": offsets.tolist(),
        "z_commanded_um": out["commanded_um"].tolist(),
        "z_readback_um": out["readback_um"].tolist(),
        "exposure_ms": exposure_ms,
        "laser_power_mW": laser_power,
        "camera_roi_xywh": list(out["roi"]),
        "dark_frames": pc["dark_frames"],
        "dark_subtracted": True,
        "elapsed_s": out["elapsed_s"],
    }
    path = save_zstack_tiff(session.path, stack.astype(np.float32), metadata, z_step_um=step)
    save_phase_mask(session.path, out["mask"], out["mask_info"])
    save_frame_tiff(session.path, out["dark"].astype(np.float32),
                    {"frames_averaged": pc["dark_frames"], "exposure_ms": exposure_ms},
                    filename="dark_mean.tif")
    logger.info(f"Dark-subtracted PSF stack ({stack.shape[0]} planes, "
                f"{offsets[0]:+.3f} to {offsets[-1]:+.3f} um) saved to {path}")


def _acquire(mmc, config, pc, slm, stage, logger, half_range, step, exposure_ms,
             laser_power, moved):
    # --- Step 1: position the guide star ---
    preview_sample_positioning(mmc, config, slm, logger, exposure_ms, laser_power)

    # --- Step 2: channel, guide-star verification, crop ---
    channels = [c for c in config.get("camera", "channel_rois", default={})
                if not c.startswith("_")]
    channel = prompt_choice("Which polarization channel should be imaged?",
                            channels or ["left", "right"])
    roi = set_channel_roi(mmc, config, channel)
    frame, loc = verify_point_source(mmc, config, slm, logger, exposure_ms,
                                     laser_power, "windowed")
    show_verification_frame(
        frame, loc, title=f"ReflEx3D — guide star, channel '{channel}' (close to continue)")
    if loc is None or not prompt_yes_no("Is this the guide star?", default=False):
        logger.info("Aborted at guide-star verification.")
        return None
    if pc.get("guide_star_roi_px"):
        roi = set_guide_star_roi(mmc, (roi[0] + loc["x"], roi[1] + loc["y"]),
                                 pc["guide_star_roi_px"])
    logger.info(f"Camera ROI for the stack: {roi}")

    # --- Step 3: phase mask ---
    while True:
        preset = prompt_choice("Phase mask for the stack (Enter = flat)?",
                               list(PRESETS), default=DEFAULT_PHASE_MASK)
        try:
            mask, mask_info = build_imaging_mask(config, preset)
            break
        except ValueError as exc:
            print(f"  {exc}")
    logger.info(describe_imaging_mask(mask_info))

    # --- Step 4: confirm the scan ---
    start_positions = stage.positions()
    start = start_positions["depth"]
    positions = scan_positions(start, half_range, step)
    overshoot = pc["overshoot_steps"] * step
    per_plane = stage.settle_s + exposure_ms / 1000.0 + 0.025
    estimate = len(positions) * per_plane + pc["dark_frames"] * (exposure_ms / 1000.0 + 0.01)
    print(f"\n{len(positions)} planes, {positions[0] - start:+.3f} to {positions[-1] - start:+.3f} um "
          f"around the current depth ({start:.4f} um on '{stage.property_for('depth')}'), "
          f"step {step} um, after overshooting to {positions[0] + overshoot - start:+.3f} um. "
          f"~{format_duration(estimate)} including {pc['dark_frames']} dark frames.")
    if not prompt_yes_no("Proceed?", default=True):
        logger.info("Aborted before acquisition at the scientist's request.")
        return None

    # --- Step 5: the stack ---
    slm.show_mask(mask)
    mmc.setExposure(exposure_ms)
    set_laser_power_mmcore(mmc, config, laser_power)
    logger.info(f"Scanning: overshoot to {positions[0] + overshoot:.4f} um, then "
                f"{positions[0]:.4f} -> {positions[-1]:.4f} um")
    out = {"channel": channel, "roi": roi, "mask": mask, "mask_info": mask_info,
           "start_positions": start_positions, "start_depth_um": start}

    def progress(i, z):
        if (i + 1) % 10 == 0 or i + 1 == len(positions):
            logger.info(f"  plane {i + 1}/{len(positions)} ({z - start:+.3f} um)")

    t0 = time.time()
    time.sleep(slm_settle_seconds(config))
    laser_on_mmcore(mmc, config)
    moved["start_depth_um"] = start
    try:
        frames, readback = scan(stage, "depth", positions, overshoot,
                                acquire=lambda: mmc.snap().astype(np.float32),
                                on_step=progress)
    finally:
        laser_off_mmcore(mmc, config)
    out.update(frames=np.stack(frames), commanded_um=positions, readback_um=readback,
               elapsed_s=time.time() - t0,
               finished_at=datetime.now().isoformat(sep=" ", timespec="seconds"))
    worst = float(np.max(np.abs(readback - positions)))
    logger.info(f"Stack done: {len(positions)} planes in {out['elapsed_s']:.1f} s; "
                f"max |read-back - commanded| depth {worst * 1000:.1f} nm")

    # --- Step 6: dark stack, same ROI, laser off ---
    logger.info(f"Taking {pc['dark_frames']} dark frames (laser off)...")
    out["dark"] = acquire_dark_frame(mmc, pc["dark_frames"])
    return out


if __name__ == "__main__":
    main()
