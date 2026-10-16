"""
functions/zernike.py

General-purpose Zernike polynomials on the unit disk.

Kept deliberately general (arbitrary n, m) rather than hardcoding the
two modes the Fourier-alignment routine needs, because the same
functions are the basis for the later adaptive-optics and
PSF-engineering work (aberration correction, EDOF mask design).

Convention: ANSI/OSA ordering, Z(n, m) with m signed.
  m > 0 -> cos(m*theta),  m < 0 -> sin(|m|*theta)
Normalization is the "unit peak-to-valley over the unit disk is not
enforced" raw form; the alignment code rescales explicitly to a
requested peak-to-valley phase, so raw form is the useful one here.
"""
from math import factorial

import numpy as np


def _radial(n, m_abs, rho):
    """Radial polynomial R_n^|m|(rho)."""
    if (n - m_abs) % 2 != 0:
        return np.zeros_like(rho)
    out = np.zeros_like(rho, dtype=float)
    for k in range((n - m_abs) // 2 + 1):
        c = ((-1) ** k * factorial(n - k)) / (
            factorial(k)
            * factorial((n + m_abs) // 2 - k)
            * factorial((n - m_abs) // 2 - k)
        )
        out += c * rho ** (n - 2 * k)
    return out


def zernike(n, m, rho, theta):
    """
    Evaluate Z(n, m) on polar coordinates rho, theta.

    Values outside the unit disk (rho > 1) are not masked here — the
    caller is responsible for applying the aperture/patch mask.
    """
    m_abs = abs(m)
    if m_abs > n:
        raise ValueError(f"Zernike requires |m| <= n, got n={n}, m={m}")
    radial = _radial(n, m_abs, rho)
    if m > 0:
        return radial * np.cos(m_abs * theta)
    if m < 0:
        return radial * np.sin(m_abs * theta)
    return radial


# --- Named modes used by the Fourier-plane alignment routine ---

#: (n, m) index for each mode name accepted by the alignment experiment.
MODE_INDICES = {
    "tilt_x": (1, 1),     # Z(1, 1)  ~ rho*cos(theta) ~ x
    "tilt_y": (1, -1),    # Z(1,-1)  ~ rho*sin(theta) ~ y
    "defocus": (2, 0),    # Z(2, 0)  ~ 2*rho^2 - 1
}


def mode_surface(mode, rho, theta):
    """
    Evaluate a named mode ('tilt_x', 'tilt_y', 'defocus') on rho, theta.
    """
    if mode not in MODE_INDICES:
        raise ValueError(
            f"Unknown mode {mode!r}. Available: {sorted(MODE_INDICES)}"
        )
    n, m = MODE_INDICES[mode]
    return zernike(n, m, rho, theta)


def is_defocus_like(mode):
    """
    True for modes whose d^2 map is expected to MINIMIZE at the Fourier
    centre (defocus), False for modes expected to MAXIMIZE there (tilt).

    This single predicate is what the map analysis keys off, so adding
    a new probe mode only means adding it here and in MODE_INDICES.
    """
    return mode == "defocus"


# --- Noll-indexed, RMS-normalized modes (adaptive optics) ---

#: Conventional names for the low Noll indices, for reports.
NOLL_NAMES = {
    1: "piston", 2: "tilt x", 3: "tilt y", 4: "defocus",
    5: "oblique astigmatism", 6: "vertical astigmatism",
    7: "vertical coma", 8: "horizontal coma",
    9: "vertical trefoil", 10: "oblique trefoil",
    11: "primary spherical",
    12: "vertical secondary astigmatism", 13: "oblique secondary astigmatism",
    14: "vertical quadrafoil", 15: "oblique quadrafoil",
    16: "horizontal secondary coma", 17: "vertical secondary coma",
    18: "oblique secondary trefoil", 19: "vertical secondary trefoil",
    20: "oblique pentafoil", 21: "vertical pentafoil",
    22: "secondary spherical",
}


def noll_to_nm(j):
    """Noll index j (1-based) -> (n, m), m > 0 for cos and m < 0 for sin terms."""
    if j < 1:
        raise ValueError(f"Noll indices start at 1, got {j}")
    n = int((-1.0 + np.sqrt(8 * (j - 1) + 1)) / 2.0)
    p = j - n * (n + 1) // 2
    k = n % 2
    m = int((p + k) / 2) * 2 - k
    if m != 0 and j % 2 == 1:
        m = -m
    return n, m


def noll_zernike(j, rho, theta):
    """
    Noll mode j, normalized to unit RMS over the unit disk, so a
    coefficient of c radians is c radians RMS of wavefront phase.
    """
    n, m = noll_to_nm(j)
    norm = np.sqrt(n + 1) if m == 0 else np.sqrt(2 * (n + 1))
    return norm * zernike(n, m, rho, theta)
