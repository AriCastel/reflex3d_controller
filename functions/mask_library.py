"""
functions/mask_library.py

The phase-mask library: designed masks dropped as files into
config/mask_library/ (next to the config file), each of which becomes an
entry of the 'SLM phase mask' list and a name load_phase_mask accepts.

A mask is a `<name>.json` file in the 'reflex3d_phase_mask' format that
the PhaseMaskOptimization repository exports (psfsim/export.py,
scripts/export_masks.py), in one or both of two forms:

* parametric ("parametric": {"type": ...}): the mask's own parameters,
  evaluated here analytically on each channel's pupil, whatever its
  radius and centre on the SLM:
    zoned_focus  concentric zones (edges `radii`, fractions of the pupil
                 radius), each focused at its own depth `depths_nm` with
                 exact cos(theta) defocus, a piston `pistons_waves` and an
                 optional focus sweep `sweeps_nm`; uses the file's
                 optical_system (na, n_medium, wavelength_nm)
    zernike      `modes`: [{"n", "m", "waves_rms"}], RMS-normalized
                 (m > 0 cosine, m < 0 sine, as functions/zernike.py)
    annular      zone edges `radii` with one constant phase per zone,
                 `phases_waves`
* raster ("raster": {"file": "<name>.npy"}): a square matrix of the
  unwrapped phase in waves across the pupil's bounding square, element
  [i, j] at x = (2j + 1)/N - 1 (SLM columns), y = (2i + 1)/N - 1 (SLM
  rows), in units of the pupil radius. It is resampled (nearest
  neighbour) onto the pupil. For free-form masks with no parameters.

When a file has both, the parametric form is used (it is exact); set
"use": "raster" in the file to force the matrix. A bare `<name>.npy`
with no .json is taken as a raster mask of that name.

Phases are in waves (1 wave = 2*pi = slm.grey_level_2pi grey levels).
Never writes to disk.
"""
import json
import warnings
from pathlib import Path

import numpy as np

from functions.zernike import zernike

LIBRARY_DIRNAME = "mask_library"
FORMAT = "reflex3d_phase_mask"
PARAMETRIC_TYPES = ("zoned_focus", "zernike", "annular")


def library_dir(config):
    """The mask library folder, next to the config file."""
    return Path(config.path).parent / LIBRARY_DIRNAME


def _read(path):
    """A library entry from a .json mask file or a bare .npy raster."""
    path = Path(path)
    if path.suffix == ".npy":
        doc = {"raster": {"file": path.name}}
    else:
        with open(path, "r") as f:
            doc = json.load(f)
        if not isinstance(doc, dict) or doc.get("format") != FORMAT:
            raise ValueError(f"not a '{FORMAT}' file")
    parametric, raster = doc.get("parametric"), doc.get("raster")
    raster_path = path.parent / raster["file"] if raster else None
    if raster_path is not None and not raster_path.exists():
        raster_path = None
    use = doc.get("use") or ("parametric" if parametric else "raster")
    if use == "parametric":
        if not parametric:
            raise ValueError("no 'parametric' section")
        if parametric.get("type") not in PARAMETRIC_TYPES:
            raise ValueError(f"unknown parametric mask type {parametric.get('type')!r} "
                             f"(known: {', '.join(PARAMETRIC_TYPES)})")
    elif use == "raster":
        if raster_path is None:
            raise ValueError("no raster matrix" + (f" (missing {raster['file']})" if raster else ""))
    else:
        raise ValueError(f"'use' must be 'parametric' or 'raster', got {use!r}")
    return {
        "name": path.stem,
        "label": doc.get("label") or path.stem,
        "description": doc.get("description", ""),
        "file": str(path),
        "kind": use,
        "parametric": parametric if use == "parametric" else None,
        "raster_file": str(raster_path) if use == "raster" else None,
        "optical_system": doc.get("optical_system") or {},
    }


def list_library(config, reserved=()):
    """
    The masks in the library, {name: entry}, sorted by name (the file
    name without its extension). A file that can't be used - unreadable,
    wrong format, or named like one of the `reserved` built-in names - is
    skipped with a warning rather than taking the others down with it.
    """
    folder = library_dir(config)
    if not folder.is_dir():
        return {}
    entries = {}
    for path in sorted(folder.glob("*.json")) + sorted(folder.glob("*.npy")):
        if path.stem in entries:  # a .npy that belongs to (or is shadowed by) a .json
            continue
        if path.suffix == ".npy" and (folder / f"{path.stem}.json").exists():
            continue  # its .json was unusable; already warned about
        try:
            if path.stem in reserved or path.stem.startswith("aberration_correction"):
                raise ValueError(f"the name '{path.stem}' is reserved - rename the file")
            entries[path.stem] = _read(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            warnings.warn(f"Phase-mask library: skipping {path}: {exc}", stacklevel=2)
    return dict(sorted(entries.items()))


# ------------------------------------------------------------ evaluation

def _zone_index(rho, radii, what):
    edges = np.asarray([float(r) for r in radii])
    if edges.size and (np.any(np.diff(edges) <= 0) or edges[0] <= 0 or edges[-1] >= 1):
        raise ValueError(f"{what} radii must be strictly increasing and inside (0, 1), got {radii}")
    return np.searchsorted(edges, rho, side="right"), edges


def zoned_focus_waves(rho, params, optical_system):
    """
    Zone k (radii[k-1] <= rho < radii[k]) carries, in waves,
        -(n / lambda) [z_k (c - 1) + (s_k / (2 dc_k)) (c - cm_k)^2] + p_k
    with c = cos(theta) = sqrt(1 - (NA rho / n)^2), cm_k and dc_k the
    zone's centre and width in c: it focuses at depth z_k (spread over
    z_k +/- s_k / 2 by the sweep). Same formula as psfsim.masks.zoned_focus.
    """
    try:
        na, n_medium, wavelength = (float(optical_system[k])
                                    for k in ("na", "n_medium", "wavelength_nm"))
    except (KeyError, TypeError, ValueError):
        raise ValueError("a zoned_focus mask needs optical_system.na, .n_medium and "
                         ".wavelength_nm in its file") from None
    zone, edges = _zone_index(rho, params["radii"], "zoned_focus")
    n_zones = edges.size + 1
    depths = np.asarray(params["depths_nm"], float)
    pistons = np.asarray(params.get("pistons_waves") or np.zeros(n_zones), float)
    sweeps = np.asarray(params.get("sweeps_nm") or np.zeros(n_zones), float)
    if not depths.size == pistons.size == sweeps.size == n_zones:
        raise ValueError(f"zoned_focus needs {n_zones} depths, pistons and sweeps "
                         f"(one per zone) for {edges.size} radii")

    def cos_theta(r):
        return np.sqrt(1.0 - np.clip(na / n_medium * r, 0.0, 1.0) ** 2)

    c = cos_theta(rho)
    c_edges = cos_theta(np.concatenate(([0.0], edges, [1.0])))  # decreasing
    zone = np.minimum(zone, n_zones - 1)
    dc = (c_edges[:-1] - c_edges[1:])[zone]
    cm = (0.5 * (c_edges[:-1] + c_edges[1:]))[zone]
    return (-(n_medium / wavelength)
            * (depths[zone] * (c - 1.0) + sweeps[zone] / (2.0 * dc) * (c - cm) ** 2)
            + pistons[zone])


def zernike_waves(rho, theta, params):
    """Sum of RMS-normalized Zernike modes, coefficients in waves RMS."""
    out = np.zeros_like(rho, dtype=float)
    for mode in params["modes"]:
        n, m = int(mode["n"]), int(mode["m"])
        norm = np.sqrt(n + 1) if m == 0 else np.sqrt(2 * (n + 1))
        out += float(mode["waves_rms"]) * norm * zernike(n, m, rho, theta)
    return out


def annular_waves(rho, params):
    """Concentric zones of constant phase (one of phases_waves per zone)."""
    zone, edges = _zone_index(rho, params["radii"], "annular")
    phases = np.asarray(params["phases_waves"], float)
    if phases.size != edges.size + 1:
        raise ValueError(f"annular needs {edges.size + 1} phases for {edges.size} radii")
    return phases[zone]


def raster_waves(x, y, raster):
    """
    The raster matrix resampled (nearest neighbour) at normalized pupil
    coordinates x (SLM columns) and y (SLM rows).
    """
    raster = np.asarray(raster)
    if raster.ndim != 2 or raster.shape[0] != raster.shape[1] or not raster.size:
        raise ValueError(f"a raster mask must be a square 2D matrix, got shape {raster.shape}")
    if not np.issubdtype(raster.dtype, np.number) or not np.all(np.isfinite(raster)):
        raise ValueError("a raster mask must be finite numbers (phase in waves)")
    n = raster.shape[0]
    col = np.clip(np.floor((x + 1.0) * n / 2.0).astype(int), 0, n - 1)
    row = np.clip(np.floor((y + 1.0) * n / 2.0).astype(int), 0, n - 1)
    return raster[row, col].astype(float)


def library_phase(entry, pupil):
    """
    The mask's phase in radians (unwrapped) over a ZernikePupil's box,
    for ZernikePupil.paint's `extra_phase`.
    """
    try:
        if entry["kind"] == "raster":
            waves = raster_waves(pupil.x, pupil.y, np.load(entry["raster_file"]))
        else:
            params = entry["parametric"]
            # Clamp the box corners (rho > 1, never drawn) so no formula leaves its domain.
            rho = np.minimum(pupil.rho, 1.0)
            if params["type"] == "zoned_focus":
                waves = zoned_focus_waves(rho, params, entry["optical_system"])
            elif params["type"] == "zernike":
                waves = zernike_waves(rho, pupil.theta, params)
            else:
                waves = annular_waves(rho, params)
    except (KeyError, TypeError, OSError) as exc:
        raise ValueError(f"Phase mask '{entry['name']}' ({entry['file']}) can't be built: "
                         f"{type(exc).__name__}: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"Phase mask '{entry['name']}' ({entry['file']}): {exc}") from exc
    return 2.0 * np.pi * waves


def na_mismatch(entry, config):
    """A warning string if the mask was designed for another NA than the pupil's, else None."""
    designed = entry["optical_system"].get("na")
    actual = config.get("fourier_plane", "NA")
    if designed is None or actual is None or abs(float(designed) - float(actual)) < 1e-6:
        return None
    return (f"'{entry['name']}' was designed for NA {designed}, but fourier_plane.NA is "
            f"{actual}: its zones won't land where they were designed to.")
