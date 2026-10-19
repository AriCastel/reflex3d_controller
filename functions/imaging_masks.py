"""
functions/imaging_masks.py

Phase masks for imaging: each polarization channel's pupil on the SLM
gets a sum of Zernike modes (Noll order, rad RMS, as in
functions/adaptive_optics.py), built from

* the stored aberration correction (config/phase_masks.json, under
  imaging_masks.aberration_correction_method and the channel), and/or
* the extended-depth-of-field (EDOF) terms in
  imaging_masks.edof_zernike_rad_rms (primary spherical, Noll 11),

added together when both are used. PRESETS names the four
combinations experiments/load_phase_mask.py and the timelapse use.

Never writes to disk.
"""
from datetime import datetime

import numpy as np

from core.config import load_phase_masks
from functions.adaptive_optics import ZernikePupil
from functions.phase_masks import flat_mask
from functions.zernike import NOLL_NAMES

#: preset -> (use the stored aberration correction, add the EDOF terms)
PRESETS = {
    "flat": (False, False),
    "aberration_correction": (True, False),
    "edof": (False, True),
    "aberration_correction_edof": (True, True),
}


def stored_correction(config, method, channel, phase_masks=None):
    """A channel's stored correction entry, or a clear error if there's none."""
    if phase_masks is None:
        phase_masks = load_phase_masks(config)
    entry = phase_masks.get(method, {}).get(channel)
    if not entry or not entry.get("coefficients_rad_rms"):
        raise ValueError(
            f"No '{method}' aberration correction stored for channel "
            f"'{channel}' in {config.phase_masks_path}. Run "
            f"experiments/aberration_correction.py on that channel (and store "
            f"the result), or store an earlier run with "
            f"experiments/store_ao_correction.py."
        )
    return entry


def build_imaging_mask(config, preset, channels=None, phase_masks=None):
    """
    Full-SLM uint8 mask for `preset` (a PRESETS key) on `channels`
    (default: imaging_masks.channels), plus a dict describing exactly
    what went into it, for logging and saving alongside the data.

    Every channel is checked before anything is built, so a missing
    correction raises before the SLM is touched.
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown phase-mask preset '{preset}'; choose one of {list(PRESETS)}")
    use_correction, use_edof = PRESETS[preset]
    settings = config.get("imaging_masks", default={})
    if channels is None:
        channels = settings.get("channels", ["left", "right"])
    method = settings.get("aberration_correction_method", "spsa")
    edof = {int(j): float(c) for j, c in settings.get("edof_zernike_rad_rms", {}).items()} \
        if use_edof else {}
    if use_correction and phase_masks is None:
        phase_masks = load_phase_masks(config)

    info = {
        "preset": preset,
        "built_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "phase_masks_file": str(config.phase_masks_path) if use_correction else None,
        "channels": {},
        "warnings": [],
    }
    terms_by_channel = {}
    for channel in channels:
        terms, source = {}, None
        if use_correction:
            entry = stored_correction(config, method, channel, phase_masks)
            for j, c in entry["coefficients_rad_rms"].items():
                terms[int(j)] = terms.get(int(j), 0.0) + float(c)
            source = {"method": method,
                      **{k: entry.get(k) for k in ("saved_on", "measured_on", "run")}}
        for j, c in edof.items():
            terms[j] = terms.get(j, 0.0) + c
        terms_by_channel[channel] = terms
        info["channels"][channel] = {
            "zernike_rad_rms": {str(j): terms[j] for j in sorted(terms)},
            "aberration_correction": source,
            "edof_zernike_rad_rms": {str(j): c for j, c in edof.items()},
        }

    mask = flat_mask(config)
    pupils = []
    for channel, terms in terms_by_channel.items():
        if not terms:
            continue
        noll = sorted(terms)
        pupil = ZernikePupil(config, channel, noll)
        pupil.paint(mask, [terms[j] for j in noll])
        info["channels"][channel]["pupil_center_px"] = list(pupil.center)
        info["channels"][channel]["pupil_radius_px"] = pupil.radius
        pupils.append(pupil)

    for i, a in enumerate(pupils):
        for b in pupils[i + 1:]:
            if np.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1]) < a.radius + b.radius:
                info["warnings"].append(
                    f"The '{a.channel}' and '{b.channel}' pupils overlap on the SLM; "
                    f"'{b.channel}' was drawn over '{a.channel}' where they do."
                )
    return mask, info


def describe_imaging_mask(info):
    """Human-readable summary of `build_imaging_mask`'s info dict."""
    lines = [f"Phase mask '{info['preset']}':"]
    for channel, ch in info["channels"].items():
        if not ch["zernike_rad_rms"]:
            lines.append(f"  {channel}: flat")
            continue
        src = ch["aberration_correction"]
        if src:
            when = src.get("measured_on") or src.get("saved_on")
            lines.append(f"  {channel}: '{src['method']}' correction"
                         + (f" measured {when}" if when else "")
                         + (f" ({src['run']})" if src.get("run") else ""))
        if ch["edof_zernike_rad_rms"]:
            lines.append(f"  {channel}: + EDOF {ch['edof_zernike_rad_rms']}")
        for j, c in ch["zernike_rad_rms"].items():
            lines.append(f"    Noll {j:>2} {NOLL_NAMES.get(int(j), ''):<32} {c:+.4f} rad RMS")
    lines += [f"  WARNING: {w}" for w in info["warnings"]]
    return "\n".join(lines)


# What this process last put on the SLM (mask, info), so a later
# acquisition - e.g. a timelapse started from the napari GUI - can record
# the mask it was taken with.
_SHOWN = {}


def remember_shown_mask(mask, info):
    _SHOWN.update(mask=mask, info=info)


def shown_mask():
    """(mask, info) last shown on the SLM by this process, or (None, None)."""
    return _SHOWN.get("mask"), _SHOWN.get("info")
