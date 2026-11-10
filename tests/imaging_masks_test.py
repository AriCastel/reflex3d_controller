"""
Phase-mask store + imaging masks against a simulated two-channel
microscope (pymmcore-plus Python devices on a UniMMCore: camera, SLM,
laser). Checks:

* config/phase_masks.json writes (method -> channel), backups, and
  that storing one channel keeps the other;
* each preset's mask, decoded back to phase, is the expected Zernike
  sum in each channel's pupil and flat everywhere else (correction +
  EDOF adds the two term by term); the annular binary presets add 0 / pi
  rings (outer radii imaging_masks.annular_binary_rho, default
  [0.76, 0.93]) to that, and bad rho vectors are refused;
* the mask library (config/mask_library/): parametric zoned_focus,
  zernike and annular files, a rasterized matrix (with a .json, forced
  with "use": "raster", and as a bare .npy) each give the expected phase,
  alone and added to the stored correction; unusable files are skipped
  with a warning; the three masks shipped in config/mask_library/ build
  on the shipped config;
* a mask name + the aberration-correction toggle builds the same mask as
  the 'aberration_correction_<mask>' preset;
* a missing stored correction raises before anything is shown;
* experiments/load_phase_mask.py finds the core running in this process
  (the napari GUI's) and puts each preset's mask on the SLM, and refuses
  with a clear error when there is no core to be found;
* the 'SLM phase mask' panel (list, toggle, load button) loads the
  selected mask with the toggle's state, picks up new library files on
  Refresh, and survives a failing load;
(The timelapse's phase mask is covered by tests/timelapse_test.py.)

    python tests/imaging_masks_test.py
"""
import json
import os
import shutil
import sys
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
from pymmcore_plus.experimental.unicore import (
    GenericDevice, SimpleCameraDevice, SLMDevice, UniMMCore,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import MicroscopeConfig, load_phase_masks
from core.file_io import load_ao_metadata, write_phase_mask_coefficients
from functions.adaptive_optics import AO_METHOD, ZernikePupil, phase_mask_entry
from functions.imaging_masks import (
    available_masks, build_imaging_mask, describe_imaging_mask, parse_preset, preset_name, presets,
)
from functions.mask_library import list_library
from functions.phase_masks import annular_binary_phase
from functions.zernike import noll_zernike

W, H, R = 320, 200, 60
CENTERS = {"left": (80.0, 100.0), "right": (235.5, 96.0)}
FLAT, G2PI = 128, 255
EDOF = {11: 1.5}
ANNULAR = [0.3, 0.6, 0.85]
CORRECTION = {
    "left": {5: 0.4, 7: -0.3, 11: 0.2},
    "right": {6: -0.5, 8: 0.25, 11: -0.1, 12: 0.15},
}

WORK = Path("/tmp/imaging_masks_test")
shutil.rmtree(WORK, ignore_errors=True)
(WORK / "config").mkdir(parents=True)
CFG = WORK / "config" / "microscope_config.json"
json.dump({
    "camera": {"channel_rois": {}},
    "slm": {"device_label": "SLM", "resolution": [W, H], "flat_value": FLAT,
            "grey_level_2pi": G2PI, "refresh_rate_hz": 1000, "settle_frames": 0},
    "laser": {"device_label": "L", "power_property": "P", "state_property": "S"},
    "fourier_plane": {"radius_px": R, "channels": {
        ch: {"center_px": list(c)} for ch, c in CENTERS.items()}},
    "acquisition_defaults": {"root_folder": str(WORK / "data")},
    "imaging_masks": {"channels": ["left", "right"], "aberration_correction_method": "spsa",
                      "edof_zernike_rad_rms": {str(j): c for j, c in EDOF.items()},
                      "annular_binary_rho": ANNULAR},
}, open(CFG, "w"))
config = MicroscopeConfig(CFG)

# ---- mask library files (written the way psfsim/export.py writes them) ----
LIB = WORK / "config" / "mask_library"
LIB.mkdir()
OPTICS = {"na": 1.1, "n_medium": 1.33, "wavelength_nm": 525.0}
ZONED = {"type": "zoned_focus", "radii": [0.45, 0.8], "depths_nm": [150.0, -1600.0, 2300.0],
         "pistons_waves": [0.0, 1.75, 0.84], "sweeps_nm": [0.0, 900.0, -1200.0]}
ZERN = {"type": "zernike", "modes": [{"n": 2, "m": 0, "waves_rms": 0.37},
                                     {"n": 4, "m": 0, "waves_rms": 0.29},
                                     {"n": 3, "m": -1, "waves_rms": 0.05}]}
RINGS = {"type": "annular", "radii": [0.5, 0.7], "phases_waves": [0.0, 0.5, 0.25]}
NRAST = 47  # odd and coprime with the pupil sizes: no SLM pixel sits exactly between two samples
_c = (2.0 * np.arange(NRAST) + 1.0) / NRAST - 1.0
RASTER = (0.8 * _c[None, :] ** 2 - 0.6 * _c[:, None] + 0.3 * _c[None, :] * _c[:, None]
          ).astype(np.float32)  # not symmetric: catches a transposed / flipped lookup


def write_lib(name, **doc):
    json.dump({"format": "reflex3d_phase_mask", "version": 1, "name": name,
               "label": f"Label of {name}", "phase_units": "waves", "optical_system": OPTICS,
               **doc}, open(LIB / f"{name}.json", "w"))


write_lib("lib_zoned", parametric=ZONED, description="three zones")
write_lib("lib_zernike", parametric=ZERN)
write_lib("lib_rings", parametric=RINGS)
np.save(LIB / "lib_raster.npy", RASTER)
write_lib("lib_raster", raster={"file": "lib_raster.npy", "pixels": NRAST})
# Both forms in one file: parametric wins unless "use": "raster".
write_lib("lib_both", parametric=RINGS, raster={"file": "lib_raster.npy", "pixels": NRAST})
write_lib("lib_forced_raster", parametric=RINGS, use="raster",
          raster={"file": "lib_raster.npy", "pixels": NRAST})
np.save(LIB / "lib_bare.npy", RASTER)  # no .json at all
LIBRARY = ["lib_bare", "lib_both", "lib_forced_raster", "lib_raster", "lib_rings",
           "lib_zernike", "lib_zoned"]


def zoned_reference(rho, theta, x, y):
    """psfsim.masks.zoned_focus, written out zone by zone (waves)."""
    k_n = OPTICS["n_medium"] / OPTICS["wavelength_nm"]
    cos = lambda r: np.sqrt(1 - (OPTICS["na"] / OPTICS["n_medium"] * np.minimum(r, 1)) ** 2)
    edges = [0.0, *ZONED["radii"], 1.0]
    out = np.zeros_like(rho)
    for k in range(3):
        sel = (rho >= edges[k]) & ((rho < edges[k + 1]) | (k == 2))
        dc, cm = cos(edges[k]) - cos(edges[k + 1]), 0.5 * (cos(edges[k]) + cos(edges[k + 1]))
        c = cos(rho[sel])
        out[sel] = -k_n * (ZONED["depths_nm"][k] * (c - 1)
                           + ZONED["sweeps_nm"][k] / (2 * dc) * (c - cm) ** 2) \
            + ZONED["pistons_waves"][k]
    return out


def zernike_reference(rho, theta, x, y):
    # Noll 4 (defocus), 11 (primary spherical), 7 (vertical coma, sin theta)
    return 0.37 * noll_zernike(4, rho, theta) + 0.29 * noll_zernike(11, rho, theta) \
        + 0.05 * noll_zernike(7, rho, theta)


def rings_reference(rho, theta, x, y):
    return np.select([rho < 0.5, rho < 0.7], [0.0, 0.5], 0.25)


def raster_reference(rho, theta, x, y):
    """Nearest raster sample, by brute-force distance to the sample centres."""
    col = np.abs(x[..., None] - _c).argmin(-1)
    row = np.abs(y[..., None] - _c).argmin(-1)
    return RASTER[row, col].astype(float)


LIB_REFERENCE = {"lib_zoned": zoned_reference, "lib_zernike": zernike_reference,
                 "lib_rings": rings_reference, "lib_both": rings_reference,
                 "lib_raster": raster_reference, "lib_forced_raster": raster_reference,
                 "lib_bare": raster_reference}

failures = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def decode(mask, channel):
    """Phase (rad, wrapped) in a channel's pupil, and the pupil's grid."""
    cx, cy = CENTERS[channel]
    yy, xx = np.mgrid[0:H, 0:W]
    rho = np.hypot(xx - cx, yy - cy) / R
    theta = np.arctan2(yy - cy, xx - cx)
    inside = rho <= 1
    phase = mask.astype(float) / G2PI * 2 * np.pi - FLAT / G2PI * 2 * np.pi
    return phase, rho, theta, inside, (xx - cx) / R, (yy - cy) / R


def annular_phase(rho, edges):
    """Annular binary phase written out ring by ring (0, pi, 0, ... from the centre)."""
    ring = sum((rho >= e).astype(int) for e in edges)
    return np.pi * (ring % 2)


def expected_phase(terms, rho, theta, annular=None):
    zern = sum(c * noll_zernike(j, rho, theta) for j, c in terms.items()) if terms else 0 * rho
    return zern + (annular_phase(rho, annular) if annular else 0)


def check_mask(mask, terms_by_channel, label, annular=None, library=None):
    """`library`: reference function (rho, theta, x, y) -> waves of a library mask."""
    outside = np.ones((H, W), bool)
    for ch, terms in terms_by_channel.items():
        phase, rho, theta, inside, x, y = decode(mask, ch)
        want = expected_phase(terms, rho, theta, annular)
        if library:
            want = want + 2 * np.pi * library(rho, theta, x, y)
        err = np.angle(np.exp(1j * (phase - want)))[inside]
        check(np.abs(err).max() < np.pi / G2PI + 1e-3,
              f"{label}: '{ch}' pupil phase matches (max err {np.abs(err).max():.4f} rad)")
        outside &= ~inside
    check(np.all(mask[outside] == FLAT), f"{label}: flat outside the pupils")


# ------------------------------------------------------------------ store
print("phase_masks.json store")
check(load_phase_masks(config) == {}, "missing file reads as empty")
try:
    build_imaging_mask(config, "aberration_correction")
    check(False, "missing correction raises")
except ValueError as exc:
    check("No 'spsa' aberration correction stored for channel 'left'" in str(exc),
          "missing correction raises a clear error")

# Store the left channel from a run's metadata, as aberration_correction.py does.
run = WORK / "ao_run_left"; run.mkdir()
meta_left = {"method": AO_METHOD, "channel": "left", "finished_at": "2026-10-14 10:00:00",
             "coefficients_rad_rms": {str(j): c for j, c in CORRECTION["left"].items()},
             "metric": {"name": "power_weighted_second_moment"},
             "metric_original_end": 3.0, "metric_corrected_end": 1.0,
             "power_ratio_corrected_over_original": 1.1,
             "fourier_center_px": list(CENTERS["left"]), "pupil_radius_px": R}
json.dump(meta_left, open(run / "ao_correction_metadata.json", "w"))
meta, folder = load_ao_metadata(run)
_, backup = write_phase_mask_coefficients(config.phase_masks_path, AO_METHOD, "left",
                                          phase_mask_entry(meta, folder))
check(backup is None, "first write makes no backup")

try:
    build_imaging_mask(config, "aberration_correction")
    check(False, "half-stored correction raises")
except ValueError as exc:
    check("'right'" in str(exc), "only the right channel missing -> raises for 'right'")

# A wrong right-channel entry, then the right one (replacing it).
write_phase_mask_coefficients(config.phase_masks_path, AO_METHOD, "right",
                              {"coefficients_rad_rms": {"5": 9.0}})
_, backup = write_phase_mask_coefficients(
    config.phase_masks_path, AO_METHOD, "right",
    {"coefficients_rad_rms": {str(j): c for j, c in CORRECTION["right"].items()}})
store = load_phase_masks(config)
check(backup is not None and backup.exists(), "rewrite backs the previous file up")
check(json.load(open(backup))[AO_METHOD]["right"]["coefficients_rad_rms"] == {"5": 9.0},
      "backup holds the previous entry")
check(store[AO_METHOD]["right"]["coefficients_rad_rms"].get("5") is None,
      "an entry is replaced, not merged")
check(store[AO_METHOD]["left"]["run"] == str(run) and
      store[AO_METHOD]["left"]["measured_on"] == "2026-10-14 10:00:00",
      "storing 'right' keeps 'left' (with its provenance)")

# ------------------------------------------------------------------ masks
print("presets")
expected = {
    "flat": {ch: {} for ch in CENTERS},
    "aberration_correction": CORRECTION,
    "edof": {ch: EDOF for ch in CENTERS},
    "aberration_correction_edof": {
        ch: {j: CORRECTION[ch].get(j, 0) + EDOF.get(j, 0) for j in set(CORRECTION[ch]) | set(EDOF)}
        for ch in CENTERS},
    "annular_binary": {ch: {} for ch in CENTERS},
    "aberration_correction_annular_binary": CORRECTION,
}
for name in LIBRARY:  # a library mask alone, and added to the stored correction
    expected[name] = {ch: {} for ch in CENTERS}
    expected[f"aberration_correction_{name}"] = CORRECTION
check(list(available_masks(config)) == ["flat", "edof", "annular_binary", *LIBRARY],
      "available masks: the built-in ones, then the library by name")
check(sorted(presets(config)) == sorted(expected),
      "presets: every mask, without and with the aberration correction")
masks = {}
for preset in presets(config):
    mask, info = build_imaging_mask(config, preset)
    masks[preset] = mask
    name, with_correction = parse_preset(preset)
    check(mask.shape == (H, W) and mask.dtype == np.uint8, f"{preset}: full-SLM uint8")
    check_mask(mask, expected[preset], preset,
               annular=ANNULAR if name == "annular_binary" else None,
               library=LIB_REFERENCE.get(name))
    check(not info["warnings"], f"{preset}: no warnings")
    check(info["preset"] == preset and info["mask"] == name
          and info["aberration_correction"] is with_correction
          and preset_name(name, with_correction) == preset,
          f"{preset}: recorded as mask '{name}', correction {with_correction}")
    # The list + toggle spelling builds the same mask as the one-word preset.
    toggled, toggled_info = build_imaging_mask(config, name, aberration_correction=with_correction)
    check(np.array_equal(toggled, mask) and toggled_info["preset"] == preset,
          f"{preset}: same mask from ('{name}', aberration_correction={with_correction})")
print(describe_imaging_mask(build_imaging_mask(config, "aberration_correction_edof")[1]))
print(describe_imaging_mask(build_imaging_mask(config, "aberration_correction_annular_binary")[1]))

print("annular binary mask")
_, info = build_imaging_mask(config, "annular_binary")
check(all(ch["annular_binary_rho"] == ANNULAR and ch["zernike_rad_rms"] == {}
          for ch in info["channels"].values()),
      "annular_binary: rho recorded in the mask info, no Zernike terms")
check("flat" not in describe_imaging_mask(info) and str(ANNULAR) in describe_imaging_mask(info),
      "annular_binary: described as a ring mask, not as flat")
check(all(ch["annular_binary_rho"] is None
          for ch in build_imaging_mask(config, "edof")[1]["channels"].values()),
      "edof: no annular rho in the info")
# The rings themselves: 0 up to the first radius, pi to the next, 0, pi at the edge.
phase, rho, _, inside, _, _ = decode(masks["annular_binary"], "left")
for lo, hi, want in ((0.0, 0.3, 0.0), (0.3, 0.6, np.pi), (0.6, 0.85, 0.0), (0.85, 1.0, np.pi)):
    ring = inside & (rho > lo + 0.03) & (rho < hi - 0.03)
    err = np.abs(np.angle(np.exp(1j * (phase[ring] - want)))).max()
    check(ring.sum() > 20 and err < np.pi / G2PI + 1e-3, f"ring {lo}-{hi}: phase {want:.2f} rad")
check(np.array_equal(annular_binary_phase(rho, ANNULAR), annular_phase(rho, ANNULAR)),
      "annular_binary_phase == the ring-by-ring construction")

# The default (no annular_binary_rho in the config) is the (0.76, 0.93) three-ring mask.
saved = config.raw["imaging_masks"].pop("annular_binary_rho")
mask_default, info_default = build_imaging_mask(config, "annular_binary")
config.raw["imaging_masks"]["annular_binary_rho"] = saved
check_mask(mask_default, expected["annular_binary"], "annular_binary default rho",
           annular=[0.76, 0.93])
check(info_default["channels"]["left"]["annular_binary_rho"] == [0.76, 0.93],
      "default rho is [0.76, 0.93]")

mask, _ = build_imaging_mask(config, "annular_binary", channels=["right"])
check_mask(mask, {"right": {}}, "annular_binary on 'right' only", annular=ANNULAR)
inside_left = decode(mask, "left")[3]
check(np.all(mask[inside_left] == FLAT), "annular_binary on 'right' only: left pupil flat")

for bad in ([], [0.5, 0.5], [0.9, 0.5], [0.0, 0.5], [0.5, 1.0], [-0.1, 0.5], "abc", None, 0.5):
    config.raw["imaging_masks"]["annular_binary_rho"] = bad
    try:
        build_imaging_mask(config, "annular_binary")
        check(False, f"rho {bad!r} is refused")
    except ValueError as exc:
        check("annular binary rho" in str(exc), f"rho {bad!r} is refused")
config.raw["imaging_masks"]["annular_binary_rho"] = saved

print("mask library")
_, info = build_imaging_mask(config, "aberration_correction_lib_zoned")
lib_info = info["library_mask"]
check(lib_info["kind"] == "parametric" and lib_info["parametric"] == ZONED
      and lib_info["optical_system"] == OPTICS and lib_info["file"] == str(LIB / "lib_zoned.json")
      and all(ch["library_mask"] == "lib_zoned" for ch in info["channels"].values()),
      "library mask: parameters, optics and file recorded in the mask info")
text = describe_imaging_mask(info)
check("Label of lib_zoned" in text and "parametric zoned_focus" in text and "'spsa' correction" in text,
      "library mask + correction: both described")
print(text)
_, info = build_imaging_mask(config, "lib_bare")
check(info["library_mask"]["kind"] == "raster"
      and info["library_mask"]["raster_file"] == str(LIB / "lib_bare.npy")
      and "rasterized matrix" in describe_imaging_mask(info)
      and "flat" not in describe_imaging_mask(info),
      "bare .npy: a raster mask, described as one")
check(build_imaging_mask(config, "lib_both")[1]["library_mask"]["kind"] == "parametric"
      and build_imaging_mask(config, "lib_forced_raster")[1]["library_mask"]["kind"] == "raster",
      "both forms in a file: parametric unless \"use\": \"raster\"")
check(build_imaging_mask(config, "edof")[1]["library_mask"] is None, "edof: no library mask in the info")
check(not np.array_equal(masks["lib_raster"], masks["lib_raster"][:, ::-1])
      and not np.array_equal(RASTER, RASTER.T),
      "(the raster checks are meaningful: the matrix isn't symmetric)")

mask, _ = build_imaging_mask(config, "lib_zoned", channels=["right"])
check_mask(mask, {"right": {}}, "lib_zoned on 'right' only", library=zoned_reference)
check(np.all(mask[decode(mask, "left")[3]] == FLAT), "lib_zoned on 'right' only: left pupil flat")

# A mask designed for another NA still loads, with a warning saved alongside it.
write_lib("lib_other_na", parametric=RINGS, optical_system={**OPTICS, "na": 1.2})
config.raw["fourier_plane"]["NA"] = 1.1
_, info = build_imaging_mask(config, "lib_other_na")
check(len(info["warnings"]) == 1 and "NA 1.2" in info["warnings"][0]
      and "WARNING" in describe_imaging_mask(info), "a mask designed for another NA warns")
check(not build_imaging_mask(config, "lib_rings")[1]["warnings"], "same NA: no warning")
(LIB / "lib_other_na.json").unlink()

# Unusable files are skipped with a warning; the rest of the library still loads.
bad = {
    "bad_json": "{ not json",
    "bad_format": json.dumps({"format": "something_else"}),
    "bad_type": json.dumps({"format": "reflex3d_phase_mask", "parametric": {"type": "cubic"}}),
    "bad_no_raster": json.dumps({"format": "reflex3d_phase_mask", "raster": {"file": "gone.npy"}}),
    "bad_empty": json.dumps({"format": "reflex3d_phase_mask"}),
    "edof": json.dumps({"format": "reflex3d_phase_mask", "parametric": RINGS}),  # reserved name
    "aberration_correction_x": json.dumps({"format": "reflex3d_phase_mask", "parametric": RINGS}),
}
for name, text in bad.items():
    (LIB / f"{name}.json").write_text(text)
np.save(LIB / "bad_json.npy", RASTER)  # doesn't rescue its broken .json
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    listed = list_library(config, reserved=("flat", "edof", "annular_binary"))
check(list(listed) == LIBRARY, "unusable library files are skipped")
check(len(caught) == len(bad) and all("skipping" in str(w.message) for w in caught),
      f"each of the {len(bad)} unusable files gets a warning")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    check(build_imaging_mask(config, "edof")[1]["library_mask"] is None
          and np.array_equal(build_imaging_mask(config, "edof")[0], masks["edof"]),
          "a library file named 'edof' doesn't replace the built-in mask")
    try:
        build_imaging_mask(config, "bad_type")
        check(False, "an unusable mask is refused")
    except ValueError as exc:
        check("Unknown phase mask 'bad_type'" in str(exc), "an unusable mask is refused by name")
for name in bad:
    (LIB / f"{name}.json").unlink()
(LIB / "bad_json.npy").unlink()

# Parameters that pass the listing but can't be evaluated raise, naming the file.
write_lib("lib_broken", parametric={**ZONED, "depths_nm": [0.0, 1.0]})
try:
    build_imaging_mask(config, "lib_broken")
    check(False, "inconsistent zoned_focus parameters are refused")
except ValueError as exc:
    check("lib_broken" in str(exc) and "one per zone" in str(exc),
          "inconsistent zoned_focus parameters are refused, naming the mask")
(LIB / "lib_broken.json").unlink()

# The masks shipped with the controller, on the shipped config.
shipped = MicroscopeConfig()
shipped_library = list_library(shipped)
check({"zoned_focus_3", "zoned_focus_4_sweep", "spherical_defocus"} <= set(shipped_library),
      "shipped library has the three candidate masks")
for name in ("zoned_focus_3", "zoned_focus_4_sweep", "spherical_defocus"):
    mask, info = build_imaging_mask(shipped, name)
    w, h = shipped["slm"]["resolution"]
    drawn = sum(int((mask[decode_box] != shipped["slm"]["flat_value"]).sum() > 1000)
                for decode_box in [np.s_[:, : w // 2], np.s_[:, w // 2:]])
    check(mask.shape == (h, w) and drawn == 2 and not info["warnings"]
          and info["library_mask"]["kind"] == "parametric",
          f"shipped '{name}': builds on both pupils of the shipped config, no warnings")

mask, _ = build_imaging_mask(config, "edof", channels=["right"])
check_mask(mask, {"right": EDOF}, "edof on 'right' only")
inside_left = decode(mask, "left")[3]
check(np.all(mask[inside_left] == FLAT), "edof on 'right' only: left pupil flat")

# The stored correction gives the same mask as the AO code's own pupil.
pupil = ZernikePupil(config, "left", sorted(CORRECTION["left"]))
ref = pupil.mask([CORRECTION["left"][j] for j in sorted(CORRECTION["left"])])
inside = decode(ref, "left")[3]
check(np.array_equal(ref[inside], masks["aberration_correction"][inside]),
      "stored correction == ZernikePupil.mask of the same coefficients")

# ------------------------------------------------------------------ scripts
print("mask scripts in a running GUI process")


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
    def __init__(self, laser):
        super().__init__(); self.laser, self._exposure, self.lit = laser, 10.0, 0
    def sensor_shape(self): return (64, 64)
    def dtype(self): return np.uint16
    def get_exposure(self): return self._exposure
    def set_exposure(self, exposure): self._exposure = exposure
    def snap(self, buffer):
        buffer[:] = 100 + 1000 * (self.laser.get_property_value("S") == "open")
        self.lit += self.laser.get_property_value("S") == "open"
        return {}


import experiments.load_phase_mask as lpm

# The "GUI": a core in this process (UniMMCore() becomes CMMCorePlus.instance(),
# as load_mmcore's core would). It is created before the refusal check, with
# no SLM loaded yet, so find_slm_core finds it instead of registering a core of
# its own - which would then stay the singleton and shadow this one.
mmc = UniMMCore()

# No SLM in the GUI's core and no Micro-Manager 2.0 either: block pycromanager
# (only that key - patch.dict on sys.modules would also drop every module
# imported meanwhile, e.g. Qt's).
_saved_pycromanager = sys.modules.get("pycromanager")
sys.modules["pycromanager"] = None
with mock.patch.object(lpm, "load_config", lambda: config):
    try:
        lpm.main("flat")
        check(False, "load_phase_mask: refuses without a core")
    except RuntimeError as exc:
        check("No Micro-Manager core with the SLM found" in str(exc),
              "load_phase_mask: refuses without a core")

slm_dev, laser = SimSLM(), SimLaser()
mmc.loadPyDevice("Cam", SimCamera(laser)); mmc.loadPyDevice("SLM", slm_dev)
mmc.loadPyDevice("L", laser)
mmc.initializeAllDevices(); mmc.setCameraDevice("Cam"); mmc.setSLMDevice("SLM")

for preset in presets(config):
    with mock.patch.object(lpm, "load_config", lambda: config):  # pycromanager still blocked
        lpm.main(preset)
    check(np.array_equal(slm_dev.shown, masks[preset]),
          f"load_phase_mask '{preset}': mask on the SLM")
    check(not list((WORK / "data").glob("*/*_load_phase_mask_*")),
          f"load_phase_mask '{preset}': no run folder written")

# Mask name + toggle (the panel), and the same on the command line.
with mock.patch.object(lpm, "load_config", lambda: config):
    info = lpm.main("lib_zoned", True)
    check(np.array_equal(slm_dev.shown, masks["aberration_correction_lib_zoned"])
          and info["preset"] == "aberration_correction_lib_zoned",
          "load_phase_mask('lib_zoned', True): mask + correction on the SLM")
    lpm.main("lib_zoned", False)
    check(np.array_equal(slm_dev.shown, masks["lib_zoned"]),
          "load_phase_mask('lib_zoned', False): mask alone")
    for argv, want in ((["x", "lib_rings", "--aberration-correction"], "aberration_correction_lib_rings"),
                       (["x", "-a", "flat"], "aberration_correction"),
                       (["x", "lib_raster"], "lib_raster")):
        with mock.patch.object(sys, "argv", argv):
            lpm.main()
        check(np.array_equal(slm_dev.shown, masks[want]), f"command line {argv[1:]}: '{want}'")
    before = slm_dev.shown
    for argv in (["x"], ["x", "no_such_mask"], ["x", "aberration_correction_no_such_mask"]):
        with mock.patch.object(sys, "argv", argv):
            try:
                lpm.main()
                check(False, f"command line {argv[1:]}: refused")
            except SystemExit as exc:
                check("lib_zoned" in str(exc) and "Label of lib_zoned" in str(exc)
                      and slm_dev.shown is before,
                      f"command line {argv[1:]}: refused, listing the masks, SLM untouched")

# ------------------------------------------------------------------ panel
print("'SLM phase mask' panel")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QCheckBox, QLabel, QListWidget, QPushButton

from functions.napari_preview import add_phase_mask_panel

app = QApplication.instance() or QApplication([])
docked = []
viewer = mock.Mock()
viewer.window.add_dock_widget = lambda widget, **kw: docked.append((widget, kw))


def gui_load(name, correction):
    with mock.patch.object(lpm, "load_config", lambda: config):
        lpm.main(name, correction)


panel = add_phase_mask_panel(
    viewer, masks=lambda: {n: (m["label"], m["description"])
                           for n, m in available_masks(config).items()},
    load=gui_load, current="Flat (no mask)")
mask_list = panel.findChild(QListWidget)
toggle = panel.findChild(QCheckBox)
buttons = {b.text(): b for b in panel.findChildren(QPushButton)}
labels = lambda: " | ".join(l.text() for l in panel.findChildren(QLabel))
names = lambda: [mask_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(mask_list.count())]
check(docked and docked[0][0] is panel and docked[0][1]["name"] == "SLM phase mask",
      "panel docked as 'SLM phase mask'")
check(set(buttons) == {"Load into SLM", "Refresh list"} and toggle.text() == "Aberration correction",
      "one list, one 'Aberration correction' toggle, one 'Load into SLM' button (+ Refresh)")
check(names() == list(available_masks(config)) and mask_list.item(0).text() == "Flat (no mask)"
      and mask_list.item(names().index("lib_zoned")).text() == "Label of lib_zoned"
      and "three zones" in mask_list.item(names().index("lib_zoned")).toolTip(),
      "list shows every mask by label, description in the tooltip")
check(mask_list.currentRow() == 0 and not toggle.isChecked() and "Flat (no mask)" in labels(),
      "starts on the first mask, toggle off, showing what is on the SLM")


def select(name):
    mask_list.setCurrentRow(names().index(name))


select("lib_zoned"); buttons["Load into SLM"].click()
check(np.array_equal(slm_dev.shown, masks["lib_zoned"]) and "Label of lib_zoned</b>" in labels(),
      "Load into SLM: the selected mask, without the correction")
toggle.setChecked(True); buttons["Load into SLM"].click()
check(np.array_equal(slm_dev.shown, masks["aberration_correction_lib_zoned"])
      and "Label of lib_zoned + aberration correction" in labels(),
      "toggle on: the selected mask + the stored correction")
select("flat"); buttons["Load into SLM"].click()
check(np.array_equal(slm_dev.shown, masks["aberration_correction"]),
      "'Flat' with the toggle on: the correction alone")
check(np.array_equal(slm_dev.shown, masks["aberration_correction"]) and toggle.isChecked(),
      "selecting a mask changes nothing on the SLM until Load is pressed")

# A mask dropped into the library while the GUI runs shows up on Refresh.
write_lib("lib_new", parametric=RINGS)
check("lib_new" not in names(), "a new library file isn't listed yet")
select("lib_rings"); buttons["Refresh list"].click()
check("lib_new" in names() and names()[mask_list.currentRow()] == "lib_rings",
      "Refresh list: new mask listed, selection kept")
toggle.setChecked(False); select("lib_new"); buttons["Load into SLM"].click()
check(np.array_equal(slm_dev.shown, masks["lib_rings"]), "the new mask loads")

# A failing load (no stored correction) leaves the SLM and the panel's record alone.
shutil.move(config.phase_masks_path, WORK / "moved.json")
toggle.setChecked(True); select("lib_zoned"); before = slm_dev.shown
buttons["Load into SLM"].click()
check(slm_dev.shown is before and "FAILED" in labels() and "Label of lib_new</b>" in labels(),
      "failing load: reported, SLM untouched, panel still names the mask that is on it")
shutil.move(WORK / "moved.json", config.phase_masks_path)
buttons["Load into SLM"].click()
check(np.array_equal(slm_dev.shown, masks["aberration_correction_lib_zoned"])
      and "FAILED" not in labels(), "and loads once the correction is there")
(LIB / "lib_new.json").unlink()

if _saved_pycromanager is None:
    del sys.modules["pycromanager"]
else:
    sys.modules["pycromanager"] = _saved_pycromanager

print()
assert not failures, failures
print("IMAGING MASKS TEST PASSED")
