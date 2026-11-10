"""
functions/imaging_masks.py

Phase masks for imaging. Each polarization channel's pupil on the SLM
gets ONE mask, chosen by name, optionally on top of the stored
aberration correction:

* "flat": nothing;
* "edof": the extended-depth-of-field Zernike terms in
  imaging_masks.edof_zernike_rad_rms (primary spherical, Noll 11);
* "annular_binary": an annular binary phase mask (0 / pi rings, outer
  radii imaging_masks.annular_binary_rho as fractions of the pupil
  radius - see functions/phase_masks.py);
* any mask in the library, config/mask_library/<name>.json (designed
  masks exported by the PhaseMaskOptimization repository, parametric or
  rasterized - see functions/mask_library.py).

The aberration correction (config/phase_masks.json, under
imaging_masks.aberration_correction_method and the channel; Zernike
modes in Noll order, rad RMS, as in functions/adaptive_optics.py) is a
toggle: with it, the mask's phase is added to the correction's.

A mask + toggle pair also has a one-word name, its *preset*, which is
what experiments/load_phase_mask.py and the timelapse take and what is
saved with the data: the mask's name, or "aberration_correction" (the
correction alone, i.e. on "flat"), or "aberration_correction_<mask>".

Never writes to disk.
"""
from datetime import datetime

import numpy as np

from core.config import load_phase_masks
from functions.adaptive_optics import ZernikePupil
from functions.mask_library import library_phase, list_library, na_mismatch
from functions.phase_masks import (
    DEFAULT_ANNULAR_RHO,
    annular_binary_phase,
    check_annular_rho,
    flat_mask,
)
from functions.zernike import NOLL_NAMES

AC = "aberration_correction"

#: Built-in mask name -> (label, description), in list order.
BUILTIN_MASKS = {
    "flat": ("Flat (no mask)", "Nothing but the aberration correction, if that is on."),
    "edof": ("EDOF (primary spherical)",
             "Zernike terms in imaging_masks.edof_zernike_rad_rms."),
    "annular_binary": ("EDOF (annular binary)",
                       "0 / pi rings, outer radii imaging_masks.annular_binary_rho."),
}


def available_masks(config):
    """
    Every mask that can be loaded, {name: {"label", "description",
    "library": entry or None}}: the built-in ones, then the library
    (config/mask_library/), re-read on every call so a file dropped in
    while the GUI runs shows up.
    """
    masks = {name: {"label": label, "description": description, "library": None}
             for name, (label, description) in BUILTIN_MASKS.items()}
    for name, entry in list_library(config, reserved=BUILTIN_MASKS).items():
        masks[name] = {"label": entry["label"], "description": entry["description"],
                       "library": entry}
    return masks


def preset_name(mask, aberration_correction=False):
    """The preset (one-word name) of a mask with or without the aberration correction."""
    if not aberration_correction:
        return mask
    return AC if mask == "flat" else f"{AC}_{mask}"


def parse_preset(preset):
    """(mask name, with aberration correction) for a preset."""
    if preset == AC:
        return "flat", True
    if isinstance(preset, str) and preset.startswith(AC + "_"):
        return preset[len(AC) + 1:], True
    return preset, False


def presets(config):
    """Every valid preset: each available mask, without and with the correction."""
    masks = list(available_masks(config))
    return masks + [preset_name(m, True) for m in masks]


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


def build_imaging_mask(config, preset, channels=None, phase_masks=None,
                       aberration_correction=False):
    """
    Full-SLM uint8 mask for `preset` on `channels` (default:
    imaging_masks.channels), plus a dict describing exactly what went
    into it, for logging and saving alongside the data.

    `preset` is a mask name (see `available_masks`) or a preset that
    already includes the correction ("aberration_correction_<mask>");
    `aberration_correction=True` adds the stored correction to a plain
    mask name - the two spellings build the same mask.

    Every channel is checked before anything is built, so a missing
    correction raises before the SLM is touched.
    """
    name, use_correction = parse_preset(preset)
    use_correction = use_correction or bool(aberration_correction)
    masks = available_masks(config)
    if name not in masks:
        raise ValueError(f"Unknown phase mask '{preset}'; choose one of {list(masks)}, "
                         f"alone or as '{AC}_<mask>' ('{AC}' = the correction alone)")
    preset = preset_name(name, use_correction)
    entry = masks[name]["library"]
    settings = config.get("imaging_masks", default={})
    if channels is None:
        channels = settings.get("channels", ["left", "right"])
    method = settings.get("aberration_correction_method", "spsa")
    edof = {int(j): float(c) for j, c in settings.get("edof_zernike_rad_rms", {}).items()} \
        if name == "edof" else {}
    annular = check_annular_rho(
        settings.get("annular_binary_rho", DEFAULT_ANNULAR_RHO)) if name == "annular_binary" else None
    if use_correction and phase_masks is None:
        phase_masks = load_phase_masks(config)

    info = {
        "preset": preset,
        "mask": name,
        "aberration_correction": use_correction,
        "built_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "phase_masks_file": str(config.phase_masks_path) if use_correction else None,
        "library_mask": {k: entry[k] for k in ("name", "label", "kind", "file", "raster_file",
                                               "parametric", "optical_system")} if entry else None,
        "channels": {},
        "warnings": [],
    }
    if entry and na_mismatch(entry, config):
        info["warnings"].append(na_mismatch(entry, config))
    terms_by_channel = {}
    for channel in channels:
        terms, source = {}, None
        if use_correction:
            stored = stored_correction(config, method, channel, phase_masks)
            for j, c in stored["coefficients_rad_rms"].items():
                terms[int(j)] = terms.get(int(j), 0.0) + float(c)
            source = {"method": method,
                      **{k: stored.get(k) for k in ("saved_on", "measured_on", "run")}}
        for j, c in edof.items():
            terms[j] = terms.get(j, 0.0) + c
        terms_by_channel[channel] = terms
        info["channels"][channel] = {
            "zernike_rad_rms": {str(j): terms[j] for j in sorted(terms)},
            "aberration_correction": source,
            "edof_zernike_rad_rms": {str(j): c for j, c in edof.items()},
            "annular_binary_rho": list(annular) if annular else None,
            "library_mask": name if entry else None,
        }

    mask = flat_mask(config)
    pupils = []
    for channel, terms in terms_by_channel.items():
        if not terms and not annular and not entry:
            continue
        noll = sorted(terms)
        pupil = ZernikePupil(config, channel, noll)
        if annular:
            extra = annular_binary_phase(pupil.rho, annular)
        else:
            extra = library_phase(entry, pupil) if entry else None
        pupil.paint(mask, [terms[j] for j in noll], extra_phase=extra)
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
        library = info.get("library_mask") if ch.get("library_mask") else None
        if not ch["zernike_rad_rms"] and not ch["annular_binary_rho"] and not library:
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
        if ch["annular_binary_rho"]:
            lines.append(f"  {channel}: + annular binary phase mask, ring outer radii "
                         f"{ch['annular_binary_rho']} x pupil radius")
        if library:
            form = (f"parametric {library['parametric']['type']}" if library["kind"] == "parametric"
                    else "rasterized matrix")
            lines.append(f"  {channel}: + '{library['label']}' ({form}, {library['file']})")
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
