"""
calibration/map_analysis.py

Estimates the Fourier-plane centre from a rastered d^2 map.

Why this is not just argmin/argmax
----------------------------------
The probe patch only affects the PSF where it overlaps the pupil.
Outside the overlap region d^2 ~ 0 for BOTH probe modes. So:

* tilt      -> d^2 peaks at the centre (max pupil coverage) and decays
               to 0 outward. argmax is roughly right, but noisy.
* defocus   -> d^2 is 0 outside, rises as the patch offset grows, then
               dips back toward 0 AT the centre (no net tilt when
               centred). The map is a ring. A global argmin lands in
               the empty background, not the centre.

What IS symmetric about the true centre in both cases is the *support*
of the map: the region where the patch overlaps the pupil at all,
i.e. a disk of radius ~ (pupil_radius + patch_radius). So the support
centroid is used as the robust primary estimate, and a mode-specific
quadratic fit is applied as a refinement only if it validates
(correct curvature sign, vertex inside the support, and close to the
geometric centroid).
"""
import numpy as np
from scipy import ndimage, optimize

from functions.zernike import is_defocus_like


def _support_mask(d2, support_fraction):
    """
    Boolean mask of map points where the patch measurably overlapped
    the pupil. Holes are filled so that the defocus map's central dip
    is included in the support, and only the largest connected blob is
    kept to reject isolated hot pixels.
    """
    valid = np.isfinite(d2)
    if not np.any(valid):
        return np.zeros_like(d2, dtype=bool), float("nan")

    vals = d2[valid]
    floor = float(np.percentile(vals, 20.0))
    peak = float(np.percentile(vals, 99.0))
    if peak <= floor:
        return np.zeros_like(d2, dtype=bool), float("nan")
    threshold = floor + support_fraction * (peak - floor)

    raw = valid & (d2 > threshold)
    if not np.any(raw):
        return raw, threshold

    filled = ndimage.binary_fill_holes(raw)
    labels, n = ndimage.label(filled)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(labels), labels, range(1, n + 1))
        keep = int(np.argmax(sizes)) + 1
        filled = labels == keep
    return filled, threshold


def _ring_crest_radius(d2, XG, YG, support, cx, cy, r_support, n_bins=12):
    """
    For the defocus mode the map is a ring: zero outside, a crest at an
    intermediate offset, and a dip back down at the centre. Only the
    region INSIDE the crest is a clean bowl, so the quadratic fit for
    the central minimum must be restricted to it.

    Finds the crest radius from the radial profile of d^2 about
    (cx, cy). Returns None if no interior crest is resolvable.
    """
    r = np.sqrt((XG - cx) ** 2 + (YG - cy) ** 2)
    use = support & np.isfinite(d2)
    if use.sum() < n_bins:
        return None
    edges = np.linspace(0.0, r_support, n_bins + 1)
    prof, centers = [], []
    for i in range(n_bins):
        sel = use & (r >= edges[i]) & (r < edges[i + 1])
        if sel.sum() == 0:
            prof.append(np.nan)
        else:
            prof.append(float(np.nanmean(d2[sel])))
        centers.append(0.5 * (edges[i] + edges[i + 1]))
    prof = np.array(prof)
    centers = np.array(centers)
    if np.all(~np.isfinite(prof)):
        return None
    crest = int(np.nanargmax(prof))
    # A crest in the very first or last bin means no resolvable interior
    # bowl (either undersampled or not actually ring-shaped).
    if crest == 0 or crest >= n_bins - 1:
        return None
    return float(centers[crest])


def _quadratic_vertex(xg, yg, z, want_minimum):
    """
    Least-squares fit z = c0 + c1 x + c2 y + c3 x^2 + c4 y^2 + c5 xy and
    return the stationary point, or None if the fit is degenerate or has
    the wrong curvature for the mode.

    Coordinates are centred and scaled before fitting for conditioning.
    """
    if xg.size < 8:
        return None
    x_mu, y_mu = float(xg.mean()), float(yg.mean())
    scale = float(max(xg.std(), yg.std()))
    if scale <= 0:
        return None
    u = (xg - x_mu) / scale
    v = (yg - y_mu) / scale

    A = np.column_stack([np.ones_like(u), u, v, u ** 2, v ** 2, u * v])
    try:
        coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
    except np.linalg.LinAlgError:
        return None
    _, c1, c2, c3, c4, c5 = coeffs

    # Hessian of the fitted quadratic in (u, v)
    H = np.array([[2.0 * c3, c5], [c5, 2.0 * c4]])
    det = np.linalg.det(H)
    trace = np.trace(H)
    if det <= 0:
        return None  # saddle / degenerate, not a clean extremum
    if want_minimum and trace <= 0:
        return None
    if (not want_minimum) and trace >= 0:
        return None

    try:
        uv = np.linalg.solve(H, np.array([-c1, -c2]))
    except np.linalg.LinAlgError:
        return None
    return float(uv[0] * scale + x_mu), float(uv[1] * scale + y_mu)


def estimate_fourier_center(
    d2_map,
    x_centers,
    y_centers,
    mode,
    support_fraction=0.15,
    inner_fraction=0.45,
):
    """
    Estimate the Fourier-plane centre in SLM pixel coordinates.

    Parameters
    ----------
    d2_map : 2D array, shape (len(y_centers), len(x_centers)).
        Squared PSF separation, in camera px^2. NaN where localization
        failed.
    x_centers, y_centers : raster coordinates in SLM px.
    mode : probe mode used ('defocus', 'tilt_x', 'tilt_y').
    support_fraction : threshold for the overlap support, as a fraction
        of the map's dynamic range above its noise floor.
    inner_fraction : fallback only. The defocus fit region is normally
        bounded by the automatically detected ring crest; this fraction
        of the support radius is used if the crest cannot be resolved.

    Returns
    -------
    dict with the chosen centre and full diagnostics.
    """
    d2 = np.asarray(d2_map, dtype=float)
    XG, YG = np.meshgrid(np.asarray(x_centers, float),
                         np.asarray(y_centers, float))

    support, threshold = _support_mask(d2, support_fraction)
    n_support = int(support.sum())

    result = {
        "mode": mode,
        "expected_extremum": "minimum" if is_defocus_like(mode) else "maximum",
        "support_threshold": threshold,
        "n_support_points": n_support,
        "n_valid_points": int(np.isfinite(d2).sum()),
        "n_failed_points": int((~np.isfinite(d2)).sum()),
        "support_mask": support,
        "center_x": float("nan"),
        "center_y": float("nan"),
        "centroid_center_x": float("nan"),
        "centroid_center_y": float("nan"),
        "refined_center_x": float("nan"),
        "refined_center_y": float("nan"),
        "support_radius_px": float("nan"),
        "method": "failed",
        "fit_radius_px": float("nan"),
        "warnings": [],
    }

    if n_support < 6:
        result["warnings"].append(
            "Too few points show any sign-dependent PSF shift. The probe "
            "patch may never have overlapped the pupil, the amplitude may "
            "be too small, or localization may be failing."
        )
        return result

    # --- Primary, geometric estimate: centroid of the overlap support ---
    cx0 = float(XG[support].mean())
    cy0 = float(YG[support].mean())
    # Equivalent radius of the support region, in SLM px
    dx_step = float(np.mean(np.diff(x_centers))) if len(x_centers) > 1 else 1.0
    dy_step = float(np.mean(np.diff(y_centers))) if len(y_centers) > 1 else 1.0
    area_px2 = n_support * dx_step * dy_step
    r_support = float(np.sqrt(area_px2 / np.pi))

    result.update({
        "centroid_center_x": cx0,
        "centroid_center_y": cy0,
        "center_x": cx0,
        "center_y": cy0,
        "support_radius_px": r_support,
        "method": "support_centroid",
    })

    # A genuine overlap region is localized. Support spread over most of
    # the SLM means the map has no structure above its own scatter —
    # usually unstable localization rather than a real measurement.
    n_valid = result["n_valid_points"]
    if n_valid and n_support > 0.7 * n_valid:
        result["warnings"].append(
            f"The overlap support covers {100 * n_support / n_valid:.0f}% of "
            "the rastered area, which is far more than a pupil should. The "
            "map is probably dominated by localization scatter rather than a "
            "real signal — treat this centre as unreliable."
        )

    # --- Mode-specific refinement ---
    want_min = is_defocus_like(mode)
    if want_min:
        # Restrict to the region inside the ring crest, so the ring's
        # outer falloff does not pollute the fit for the central dip.
        r = np.sqrt((XG - cx0) ** 2 + (YG - cy0) ** 2)
        r_crest = _ring_crest_radius(d2, XG, YG, support, cx0, cy0, r_support)
        if r_crest is None:
            r_crest = inner_fraction * r_support
            result["warnings"].append(
                "Could not resolve the ring crest in the defocus map; using "
                f"a fixed inner fraction ({inner_fraction}) of the support "
                "radius for the fit region. A finer raster step would help."
            )
        result["fit_radius_px"] = float(r_crest)
        # Keep at least a couple of grid steps of data in the fit.
        r_fit = max(r_crest, 2.0 * max(dx_step, dy_step))
        fit_mask = support & np.isfinite(d2) & (r <= r_fit)
    else:
        fit_mask = support & np.isfinite(d2)

    vertex = None
    if fit_mask.sum() >= 8:
        vertex = _quadratic_vertex(
            XG[fit_mask], YG[fit_mask], d2[fit_mask], want_minimum=want_min
        )

    if vertex is None:
        result["warnings"].append(
            "Quadratic refinement did not produce a valid extremum of the "
            "expected sign; falling back to the support centroid."
        )
        return result

    vx, vy = vertex
    result["refined_center_x"] = vx
    result["refined_center_y"] = vy

    # Validate: the refined vertex must stay near the geometric centre.
    offset = np.hypot(vx - cx0, vy - cy0)
    if offset > 0.5 * r_support:
        result["warnings"].append(
            f"Quadratic vertex sits {offset:.0f} px from the support centroid "
            f"(support radius {r_support:.0f} px); rejected as unreliable and "
            "falling back to the support centroid."
        )
        return result

    result["center_x"] = vx
    result["center_y"] = vy
    result["method"] = "quadratic_fit"
    return result


def consistency_check(result, config):
    """
    Cross-check the measured support radius against the known pupil
    radius from the config. The support should be a disk of radius
    ~ (pupil_radius + patch_radius), so a large mismatch usually means
    the configured radius is wrong or the probe amplitude is off.
    Appends any complaint to result['warnings'].
    """
    r_pupil = config.get("fourier_plane", "radius_px")
    r_support = result.get("support_radius_px")
    patch_d = result.get("patch_diameter_px")
    if not r_pupil or not np.isfinite(r_support or float("nan")):
        return result
    expected = r_pupil + (patch_d / 2.0 if patch_d else 0.0)
    if expected <= 0:
        return result
    # The lower bound is generous on purpose: for tilt, d^2 falls below
    # the support threshold well before the patch geometrically stops
    # overlapping the pupil, so a support noticeably smaller than
    # (R + r_patch) is expected rather than suspicious.
    ratio = r_support / expected
    if ratio < 0.35 or ratio > 2.0:
        result["warnings"].append(
            f"Measured overlap support radius ({r_support:.0f} px) differs a "
            f"lot from the value implied by the configured pupil radius "
            f"({expected:.0f} px). Check fourier_plane.radius_px and the "
            "probe amplitude."
        )
    return result


# ---------------------------------------------------------------------
# Refinement of an existing estimate from a small, fine, windowed raster
# ---------------------------------------------------------------------
#
# estimate_fourier_center() is not designed for a map that only covers a
# disk around the previous estimate: its primary estimate is the centroid
# of the overlap support, and when the whole window lies inside the
# support that is just the window centre - i.e. the old estimate back -
# leaving only its quadratic refinement, whose validity checks assume the
# full-SLM support.
#
# What survives windowing is the *shape* of the response. For a circular
# patch on a circular pupil, d^2 depends only on the distance between the
# patch centre and the pupil centre (for tilt: a plateau then a fall-off;
# for defocus: a dip inside a ring). So the pupil centre is the point
# about which the map is best described as a function of radius alone.

def _hat_basis(r, knots):
    """Piecewise-linear ("hat") basis evaluated at radii r, shape (n, K)."""
    idx = np.clip(np.searchsorted(knots, r) - 1, 0, len(knots) - 2)
    t = np.clip((r - knots[idx]) / (knots[idx + 1] - knots[idx]), 0.0, 1.0)
    B = np.zeros((r.size, len(knots)))
    rows = np.arange(r.size)
    B[rows, idx] = 1.0 - t
    B[rows, idx + 1] = t
    return B


def _radial_residual(center, xg, yg, z, knots):
    """Sum of squared residuals of the best radial profile about `center`."""
    r = np.hypot(xg - center[0], yg - center[1])
    B = _hat_basis(r, knots)
    coef, *_ = np.linalg.lstsq(B, z, rcond=None)
    res = z - B @ coef
    return float(res @ res)


def refine_fourier_center(
    d2_map,
    x_centers,
    y_centers,
    mode,
    previous_center,
    window_radius_px,
    search_radius_px=None,
    n_knots=10,
    min_explained=0.5,
):
    """
    Refine a Fourier-plane centre from a fine raster of a disk around it.

    Finds the centre c minimising the residual of a free-form radial
    profile d2(|p - c|) fitted to the map (a coarse grid over the search
    area picks the basin, Nelder-Mead polishes it).

    Parameters
    ----------
    d2_map : 2D array (len(y_centers), len(x_centers)); NaN where not
        acquired or where localization failed.
    previous_center : (x, y) estimate the window was centred on.
    window_radius_px : radius of the rastered disk.
    search_radius_px : how far from `previous_center` the new centre may
        lie; defaults to half the window radius.
    min_explained : below this fraction of the map's variance explained
        by the radial model, the result is flagged as unreliable.

    Returns a dict shaped like estimate_fourier_center()'s, plus
    previous_center_x/y, shift_px, explained_fraction.
    """
    d2 = np.asarray(d2_map, dtype=float)
    XG, YG = np.meshgrid(np.asarray(x_centers, float),
                         np.asarray(y_centers, float))
    px, py = float(previous_center[0]), float(previous_center[1])
    in_window = np.hypot(XG - px, YG - py) <= window_radius_px + 1e-9
    valid = np.isfinite(d2) & in_window
    if search_radius_px is None:
        search_radius_px = 0.5 * window_radius_px

    result = {
        "mode": mode,
        "expected_extremum": "minimum" if is_defocus_like(mode) else "maximum",
        "support_threshold": float("nan"),
        "n_support_points": 0,
        "n_valid_points": int(valid.sum()),
        "n_failed_points": int((in_window & ~np.isfinite(d2)).sum()),
        "support_mask": valid,
        "center_x": float("nan"),
        "center_y": float("nan"),
        "centroid_center_x": float("nan"),
        "centroid_center_y": float("nan"),
        "refined_center_x": float("nan"),
        "refined_center_y": float("nan"),
        "support_radius_px": float("nan"),
        "method": "failed",
        "fit_radius_px": float("nan"),
        "previous_center_x": px,
        "previous_center_y": py,
        "window_radius_px": float(window_radius_px),
        "search_radius_px": float(search_radius_px),
        "shift_px": float("nan"),
        "explained_fraction": float("nan"),
        "warnings": [],
    }

    if valid.sum() < 3 * n_knots:
        result["warnings"].append(
            f"Only {int(valid.sum())} valid map points - too few to fit a "
            "radial profile. Check localization (laser power, exposure)."
        )
        return result

    xg, yg, z = XG[valid], YG[valid], d2[valid]
    ss_tot = float(((z - z.mean()) ** 2).sum())
    if ss_tot <= 0:
        result["warnings"].append("The map is constant; nothing to fit.")
        return result

    knots = np.linspace(0.0, window_radius_px + search_radius_px, n_knots)

    def cost(c):
        return _radial_residual(c, xg, yg, z, knots)

    # Coarse grid over the search disk to pick the basin, then polish.
    n_grid = 9
    offs = np.linspace(-search_radius_px, search_radius_px, n_grid)
    best, best_c = np.inf, (px, py)
    for dy in offs:
        for dx in offs:
            if np.hypot(dx, dy) > search_radius_px:
                continue
            c = (px + dx, py + dy)
            v = cost(c)
            if v < best:
                best, best_c = v, c
    step0 = max(2.0 * search_radius_px / (n_grid - 1), 1.0)
    opt = optimize.minimize(
        cost, best_c, method="Nelder-Mead",
        options={"initial_simplex": np.array(best_c) + step0 * np.array(
                     [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
                 "xatol": 0.05, "fatol": 1e-9 * ss_tot, "maxiter": 400},
    )
    cx, cy = float(opt.x[0]), float(opt.x[1])
    ss_res = float(opt.fun)

    shift = float(np.hypot(cx - px, cy - py))
    explained = 1.0 - ss_res / ss_tot
    result.update({
        "center_x": cx,
        "center_y": cy,
        "refined_center_x": cx,
        "refined_center_y": cy,
        "centroid_center_x": px,
        "centroid_center_y": py,
        "method": "radial_symmetry",
        "shift_px": shift,
        "explained_fraction": explained,
    })

    if explained < min_explained:
        result["warnings"].append(
            f"A radial profile explains only {100 * explained:.0f}% of the "
            "map's variance, so there is no clear symmetry centre. The "
            "window may be too small or too far off the pupil, the probe "
            "amplitude too weak, or localization too noisy."
        )
    if shift > 0.9 * search_radius_px:
        result["warnings"].append(
            f"The new centre is {shift:.0f} px from the previous one, at the "
            f"edge of the {search_radius_px:.0f} px search area - the true "
            "centre may lie beyond it. Re-run fourier_plane_alignment for a "
            "fresh coarse estimate instead."
        )
    n_expected = int(in_window.sum())
    if result["n_failed_points"] > 0.3 * n_expected:
        result["warnings"].append(
            f"{result['n_failed_points']}/{n_expected} positions failed to "
            "localize; the map is probably unreliable."
        )
    return result
