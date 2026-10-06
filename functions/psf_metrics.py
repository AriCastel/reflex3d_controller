"""
functions/psf_metrics.py

Guide-star PSF quality metrics for the adaptive-optics optimizer.
Every metric follows the same convention: LOWER IS BETTER, so the
optimizer always minimizes.

A metric is a small class:

    class MyMetric(PSFMetric):
        def __init__(self, some_param=1.0):
            self.some_param = some_param
        def __call__(self, image):
            return float(...)

Register it in METRICS and select it in the config with
`adaptive_optics.metric = {"name": "my_metric", "params": {...}}`.

Images are background-subtracted guide-star crops (float, negatives
clipped to 0). `set_reference(image)` is called once with the
uncorrected (flat-mask) PSF before optimizing, so metrics can compare
against it - which is how the power-aware metrics stop the optimizer
from "improving" the PSF by throwing light away.

Never writes to disk.
"""
import numpy as np


def total_power(image):
    return float(np.sum(image))


def centroid(image, threshold_fraction=0.25):
    """Intensity centroid (x, y) of the pixels above a fraction of the peak."""
    peak = float(image.max())
    if peak <= 0:
        h, w = image.shape
        return (w - 1) / 2.0, (h - 1) / 2.0
    weights = np.where(image > threshold_fraction * peak, image, 0.0)
    yy, xx = np.indices(image.shape)
    total = weights.sum()
    return float((weights * xx).sum() / total), float((weights * yy).sum() / total)


def _radius_map(image):
    cx, cy = centroid(image)
    yy, xx = np.indices(image.shape)
    return np.hypot(xx - cx, yy - cy)


class PSFMetric:
    """Base class. Subclasses implement __call__(image) -> float."""

    def set_reference(self, image):
        """Called once with the uncorrected PSF. No-op unless overridden."""

    def __call__(self, image):
        raise NotImplementedError


class SecondMoment(PSFMetric):
    """
    Intensity-weighted mean squared radius about the centroid, within
    `radius_px` (the previous implementation's metric). Normalized by
    the enclosed power, so it does NOT conserve power: a dimmer but
    tighter PSF scores better.
    """

    def __init__(self, radius_px=15):
        self.radius_px = radius_px

    def __call__(self, image):
        r = _radius_map(image)
        inside = r <= self.radius_px
        power = image[inside].sum()
        if power <= 0:
            return float("inf")
        return float((image[inside] * r[inside] ** 2).sum() / power)


class PowerWeightedSecondMoment(SecondMoment):
    """
    SecondMoment, multiplied by (reference power / current power) ** w.
    Losing light now costs as much as a wider PSF, so the optimizer
    can't settle in a dim local minimum. w = 0 recovers SecondMoment.
    """

    def __init__(self, radius_px=15, power_weight=1.0):
        super().__init__(radius_px)
        self.power_weight = power_weight
        self.reference_power = None

    def set_reference(self, image):
        self.reference_power = total_power(image)

    def __call__(self, image):
        moment = super().__call__(image)
        power = total_power(image)
        if power <= 0:
            return float("inf")
        ref = self.reference_power or power
        return moment * (ref / power) ** self.power_weight


class EncircledEnergy(PSFMetric):
    """
    Minus the fraction of the reference PSF's total power that falls
    within `radius_px` of the centroid. Rewards concentrating light
    and, being normalized to the reference rather than the current
    image, penalizes losing it.
    """

    def __init__(self, radius_px=4):
        self.radius_px = radius_px
        self.reference_power = None

    def set_reference(self, image):
        self.reference_power = total_power(image)

    def __call__(self, image):
        ref = self.reference_power or total_power(image) or 1.0
        inside = _radius_map(image) <= self.radius_px
        return -float(image[inside].sum()) / ref


class Sharpness(PSFMetric):
    """
    Minus sum(I^2), normalized by the reference power squared. The
    classic sharpness metric divides by the current power instead,
    which makes it power-blind; this version keeps power in it.
    """

    def __init__(self):
        self.reference_power = None

    def set_reference(self, image):
        self.reference_power = total_power(image)

    def __call__(self, image):
        ref = self.reference_power or total_power(image) or 1.0
        return -float(np.sum(image.astype(float) ** 2)) / ref ** 2


METRICS = {
    "second_moment": SecondMoment,
    "power_weighted_second_moment": PowerWeightedSecondMoment,
    "encircled_energy": EncircledEnergy,
    "sharpness": Sharpness,
}


def make_metric(name, params=None):
    """Build a registered metric from its config name and params."""
    if name not in METRICS:
        raise ValueError(f"Unknown PSF metric {name!r}. Available: {sorted(METRICS)}")
    return METRICS[name](**(params or {}))
