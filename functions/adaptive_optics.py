"""
functions/adaptive_optics.py

Sensorless adaptive optics on a guide-star PSF: the SLM phase mask is
a sum of Zernike modes over the channel's pupil, and the mode
coefficients are optimized by SPSA (simultaneous perturbation
stochastic approximation) - a stochastic gradient descent that
estimates the gradient along ONE random +/-1 direction per iteration
from just two measurements, F(a + c*delta) and F(a - c*delta),
however many modes are corrected.

Speed matters because the guide star photobleaches while this runs:

* only 2 snaps per iteration (the metric at the current point is
  estimated as their mean instead of costing a third snap);
* the camera is cropped to a small ROI around the guide star, which
  shortens readout;
* the Zernike basis is precomputed once, over the pupil only, so
  building a mask is one tensor contraction.

The optimizer (`spsa_minimize`) only sees an `evaluate(coefficients)
-> metric` callable, so it is hardware-free, and the metric is any
functions/psf_metrics.py metric. Never writes to disk.
"""
import time

import numpy as np

from functions.phase_masks import flat_mask, phase_to_grey
from functions.zernike import NOLL_NAMES, noll_zernike


# --------------------------------------------------------------- masks

class ZernikePupil:
    """
    Zernike modes registered to one channel's pupil on the SLM (centre
    from fourier_plane.channels.<channel>.center_px, radius from
    fourier_plane.radius_px), turning coefficients into SLM masks.
    """

    def __init__(self, config, channel, noll_indices):
        center = config.get("fourier_plane", "channels", channel, "center_px")
        if center is None or None in center:
            raise ValueError(
                f"No Fourier-plane centre for channel '{channel}' - run "
                f"experiments/fourier_plane_alignment.py for it first."
            )
        self.channel = channel
        self.noll_indices = list(noll_indices)
        self.center = (float(center[0]), float(center[1]))
        self.radius = float(config.get("fourier_plane", "radius_px"))
        self.grey_2pi = config.get("slm", "grey_level_2pi", default=255)
        self.flat = flat_mask(config)
        self.flat_phase = (
            config.get("slm", "flat_value", default=128) / self.grey_2pi * 2 * np.pi
        )

        height, width = self.flat.shape
        cx, cy = self.center
        x0, x1 = max(0, int(np.floor(cx - self.radius))), min(width, int(np.ceil(cx + self.radius)) + 1)
        y0, y1 = max(0, int(np.floor(cy - self.radius))), min(height, int(np.ceil(cy + self.radius)) + 1)
        self.box = (slice(y0, y1), slice(x0, x1))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        rho = np.hypot(xx - cx, yy - cy) / self.radius
        theta = np.arctan2(yy - cy, xx - cx)
        self.inside = rho <= 1.0
        self.basis = np.stack(
            [np.where(self.inside, noll_zernike(j, rho, theta), 0.0) for j in self.noll_indices]
        ).astype(np.float32)

    def phase(self, coefficients):
        """Pupil-box phase map in radians (NaN outside the pupil)."""
        phase = np.tensordot(np.asarray(coefficients, np.float32), self.basis, axes=1)
        return np.where(self.inside, phase, np.nan)

    def mask(self, coefficients):
        """Full-SLM uint8 mask: flat background, Zernike phase in the pupil."""
        phase = np.tensordot(np.asarray(coefficients, np.float32), self.basis, axes=1)
        mask = self.flat.copy()
        region = mask[self.box]
        region[self.inside] = phase_to_grey(phase[self.inside] + self.flat_phase, self.grey_2pi)
        return mask


# --------------------------------------------------------------- camera

def set_guide_star_roi(mmc, center_xy, size_px):
    """
    Crop the camera to a square around the guide star (full-frame
    coordinates), clamped to the sensor. Returns the ROI the camera
    actually applied, which some cameras round to their granularity.
    """
    mmc.clearROI()
    sensor_w, sensor_h = mmc.getImageWidth(), mmc.getImageHeight()
    size = int(min(size_px, sensor_w, sensor_h))
    x = int(round(center_xy[0] - size / 2))
    y = int(round(center_xy[1] - size / 2))
    x = min(max(x, 0), sensor_w - size)
    y = min(max(y, 0), sensor_h - size)
    mmc.setROI(x, y, size, size)
    return tuple(mmc.getROI())


def acquire_dark_frame(mmc, n_frames):
    """Average of `n_frames` snaps; call with the illumination OFF."""
    total = None
    for _ in range(n_frames):
        frame = mmc.snap().astype(np.float32)
        total = frame if total is None else total + frame
    return total / n_frames


def snap_guide_star(mmc, dark):
    """One dark-subtracted guide-star frame (float, negatives clipped)."""
    return np.clip(mmc.snap().astype(np.float32) - dark, 0, None)


# --------------------------------------------------------------- optimizer

def spsa_minimize(
    evaluate,
    n_params,
    iterations,
    gain,
    perturbation,
    momentum=0.0,
    max_abs=None,
    gain_decay=0.602,
    perturbation_decay=0.101,
    stability=None,
    patience=None,
    rng=None,
    on_iteration=None,
):
    """
    Minimize `evaluate(params) -> float` by SPSA, starting from zero.

    Per iteration k, with delta a random +/-1 vector:
        c_k  = perturbation / (k + 1) ** perturbation_decay
        g    = (F(a + c_k delta) - F(a - c_k delta)) / (2 c_k |F(0)|) * delta
        v    = momentum * v + a_k * g,   a_k = gain * ((1 + A) / (k + 1 + A)) ** gain_decay
        a   <- clip(a - v, -max_abs, max_abs)
    The gradient is divided by |F(0)| so `gain` doesn't depend on the
    metric's units. A (`stability`) defaults to 10% of `iterations`.

    `patience`: stop early after this many iterations without the
    metric improving on its best value by 1% of |F(0)| (None = never).

    Returns (params, history, initial_metric), history holding the
    per-iteration metric estimate (mean of the two evaluations) and
    the params after each update.
    """
    rng = np.random.default_rng(rng)
    A = 0.1 * iterations if stability is None else stability
    params = np.zeros(n_params)
    velocity = np.zeros(n_params)

    initial = float(evaluate(params))
    scale = abs(initial) or 1.0
    history = {"iteration": [], "metric": [], "params": []}
    best, since_best = initial, 0

    for k in range(iterations):
        a_k = gain * ((1 + A) / (k + 1 + A)) ** gain_decay
        c_k = perturbation / (k + 1) ** perturbation_decay
        delta = rng.choice([-1.0, 1.0], size=n_params)

        m_plus = float(evaluate(params + c_k * delta))
        m_minus = float(evaluate(params - c_k * delta))
        grad = (m_plus - m_minus) / (2 * c_k * scale) * delta

        velocity = momentum * velocity + a_k * grad
        params = params - velocity
        if max_abs is not None:
            params = np.clip(params, -max_abs, max_abs)

        metric = 0.5 * (m_plus + m_minus)
        history["iteration"].append(k + 1)
        history["metric"].append(metric)
        history["params"].append(params.copy())
        if on_iteration is not None:
            on_iteration(k + 1, metric, params)

        if metric < best - 0.01 * scale:
            best, since_best = metric, 0
        else:
            since_best += 1
        if patience is not None and since_best >= patience:
            break

    return params, history, initial


# --------------------------------------------------------------- experiment

def run_aberration_correction(mmc, config, slm, pupil, metric, dark, logger,
                              settle_s, iterations, gain, perturbation,
                              momentum=0.0, max_abs=None, patience=None,
                              seed=None, progress_every=10,
                              laser_on=None, laser_off=None,
                              blank_laser_during_settle=False):
    """
    Optimize the pupil's Zernike coefficients on the guide star.

    The camera must already be cropped to the guide star and the
    laser ON. If `blank_laser_during_settle`, `laser_off`/`laser_on`
    are called around every SLM settle, so the bead is only lit while
    the camera is exposing (only worth it if the laser switches much
    faster than the settle time).

    After optimizing, the flat and the corrected mask are snapped back
    to back, so the comparison isn't skewed by the bleaching that
    happened during the run.
    """
    n_evaluations = 0

    def show(coefficients):
        if blank_laser_during_settle:
            laser_off()
        slm.show_mask(pupil.mask(coefficients))
        time.sleep(settle_s)
        if blank_laser_during_settle:
            laser_on()

    def evaluate(coefficients):
        nonlocal n_evaluations
        show(coefficients)
        n_evaluations += 1
        return metric(snap_guide_star(mmc, dark))

    zeros = np.zeros(len(pupil.noll_indices))
    show(zeros)
    reference = snap_guide_star(mmc, dark)
    metric.set_reference(reference)

    t_start = time.time()

    def on_iteration(k, value, params):
        if progress_every and k % progress_every == 0:
            rate = (time.time() - t_start) / k
            logger.info(f"  iteration {k}/{iterations}: metric {value:.4g} "
                        f"({rate * 1000:.0f} ms/iteration)")

    coefficients, history, initial = spsa_minimize(
        evaluate, len(pupil.noll_indices), iterations, gain, perturbation,
        momentum=momentum, max_abs=max_abs, patience=patience, rng=seed,
        on_iteration=on_iteration,
    )
    elapsed = time.time() - t_start
    logger.info(f"Optimization finished: {len(history['iteration'])} iterations, "
                f"{n_evaluations} snaps in {elapsed:.1f} s")

    show(zeros)
    original = snap_guide_star(mmc, dark)
    show(coefficients)
    corrected = snap_guide_star(mmc, dark)
    metric_original, metric_corrected = metric(original), metric(corrected)
    slm.show_mask(pupil.flat)

    return {
        "coefficients": coefficients,
        "history": history,
        "metric_initial": initial,
        "metric_original": metric_original,
        "metric_corrected": metric_corrected,
        "power_ratio": float(corrected.sum() / max(original.sum(), 1e-12)),
        "reference_psf": reference,
        "original_psf": original,
        "corrected_psf": corrected,
        "phase": pupil.phase(coefficients),
        "elapsed_s": elapsed,
        "n_snaps": n_evaluations + 3,
    }


def coefficient_table(noll_indices, coefficients):
    """Human-readable table of the corrected Zernike coefficients."""
    lines = [f"{'Noll':>4}  {'mode':<32} {'coeff (rad RMS)':>15}"]
    for j, c in zip(noll_indices, coefficients):
        lines.append(f"{j:>4}  {NOLL_NAMES.get(j, ''):<32} {c:>15.4f}")
    return "\n".join(lines)


def build_ao_figure(result, noll_indices, metric_name, channel):
    """Metric vs iteration, original vs corrected PSF, final mask, coefficients."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 8.5))
    grid = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(grid[0, :2])
    hist = result["history"]
    ax.plot(hist["iteration"], hist["metric"], "-", color="#1f6fb2", lw=1.2,
            label="metric (mean of +/- evaluations)")
    ax.axhline(result["metric_initial"], color="0.5", ls="--", lw=1, label="flat mask, start")
    last = hist["iteration"][-1] if hist["iteration"] else 0
    ax.plot([last], [result["metric_corrected"]], "o", color="#c0392b",
            label="corrected mask, end")
    ax.plot([last], [result["metric_original"]], "s", color="0.4",
            label="flat mask, end")
    ax.set_xlabel("iteration")
    ax.set_ylabel(f"{metric_name} (lower is better)")
    ax.set_title("Aberration metric")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(grid[0, 2])
    phase = result["phase"]
    lim = float(np.nanmax(np.abs(phase))) or 1.0
    im = ax.imshow(phase, cmap="RdBu_r", vmin=-lim, vmax=lim)
    fig.colorbar(im, ax=ax, label="phase (rad)")
    ax.set_title("Final correction phase (pupil)")
    ax.set_xticks([]); ax.set_yticks([])

    original, corrected = result["original_psf"], result["corrected_psf"]
    vmax = float(max(original.max(), corrected.max()))
    for col, (img, title) in enumerate([
        (original, f"Original PSF (metric {result['metric_original']:.4g})"),
        (corrected, f"Corrected PSF (metric {result['metric_corrected']:.4g})"),
    ]):
        ax = fig.add_subplot(grid[1, col])
        im = ax.imshow(img, cmap="magma", vmin=0, vmax=vmax)
        fig.colorbar(im, ax=ax, label="counts (dark-subtracted)")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(grid[1, 2])
    ax.bar([str(j) for j in noll_indices], result["coefficients"], color="#1f6fb2")
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xlabel("Noll index")
    ax.set_ylabel("coefficient (rad RMS)")
    ax.set_title("Zernike coefficients")

    fig.suptitle(
        f"Aberration correction — channel '{channel}'   "
        f"[corrected/original power {result['power_ratio']:.2f}]",
        fontsize=12,
    )
    fig.tight_layout()
    return fig
