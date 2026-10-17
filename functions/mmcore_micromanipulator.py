"""
functions/mmcore_micromanipulator.py

The Sutter MPC micromanipulator the sample is mounted on, addressed
through a CMMCorePlus instance. Never writes to disk.

Thanks to Luis Bastidas, who first worked out how to drive this
micromanipulator reliably and wrote the original Z-stack code this is
based on.

Three things about this stage shape the code:

* Its axes are named from the stage's point of view, and the stage is
  sometimes physically re-mounted for an experiment, so 'Current X'
  isn't necessarily horizontal on the camera, nor 'Current Z' the
  objective-sample distance. micromanipulator.axis_properties in the
  config maps the camera's 'horizontal', 'vertical' and 'depth' axes to
  the stage properties, and everything here works in those camera axes.
* Positions are set by writing the 'Current X/Y/Z' properties directly,
  which is the most reliable way to move this stage (rather than a
  speed or move-by property).
* Its microstep is 62.5 nm, but mechanical backlash means a position is
  only reproducible when it's approached from the same direction. Scans
  therefore overshoot past their first position and only ever move in
  one direction (see `approach` and `scan`).
"""
import time

import numpy as np

CAMERA_AXES = ("horizontal", "vertical", "depth")


class Micromanipulator:
    """
    Usage:
        stage = Micromanipulator(mmc, config)
        z0 = stage.position("depth")
        stage.move_to("depth", z0 - 1.0)
    """

    def __init__(self, mmc, config):
        self.mmc = mmc
        mm = config.get("micromanipulator", default={})
        self.label = mm.get("device_label", "Sutter MPC")
        self.properties = dict(mm.get("axis_properties", {}))
        missing = [a for a in CAMERA_AXES if not self.properties.get(a)]
        if missing:
            raise KeyError(
                f"micromanipulator.axis_properties is missing {missing} - map each of "
                f"{list(CAMERA_AXES)} to the stage property ('Current X', ...) that moves "
                f"along it."
            )
        if len(set(self.properties[a] for a in CAMERA_AXES)) != len(CAMERA_AXES):
            raise ValueError(
                f"micromanipulator.axis_properties maps two camera axes to the same "
                f"stage property: {self.properties}"
            )
        self.microstep_um = float(mm.get("microstep_um", 0.0625))
        self.settle_s = float(mm.get("settle_s", 0.2))

    def property_for(self, axis):
        if axis not in CAMERA_AXES:
            raise ValueError(f"axis must be one of {CAMERA_AXES}, got '{axis}'")
        return self.properties[axis]

    def position(self, axis):
        """Current position along a camera axis, in um, as the stage reports it."""
        return float(self.mmc.getProperty(self.label, self.property_for(axis)))

    def positions(self):
        """{camera axis: position in um} for all three axes."""
        return {axis: self.position(axis) for axis in CAMERA_AXES}

    def move_to(self, axis, position_um, settle_s=None):
        """
        Set the stage property for `axis` to `position_um`, wait for the
        device, then for `settle_s` (default micromanipulator.settle_s).
        """
        self.mmc.setProperty(self.label, self.property_for(axis), f"{position_um:.4f}")
        self.mmc.waitForDevice(self.label)
        time.sleep(self.settle_s if settle_s is None else settle_s)

    def check_step(self, step_um):
        """Raise unless `step_um` is a positive whole number of microsteps."""
        n = step_um / self.microstep_um
        if step_um <= 0 or abs(n - round(n)) > 1e-6:
            raise ValueError(
                f"Step {step_um} um isn't a whole number of {self.microstep_um} um "
                f"microsteps - use e.g. {max(1, round(n)) * self.microstep_um} um."
            )


def scan_positions(center_um, half_range_um, step_um):
    """
    Positions from center + half_range DOWN to center - half_range in
    steps of `step_um` (descending, so the scan only ever moves one way),
    always including the centre. half_range is rounded down to a whole
    number of steps.
    """
    n = int(np.floor(half_range_um / step_um + 1e-9))
    return center_um + step_um * np.arange(n, -n - 1, -1)


def approach(stage, axis, target_um, overshoot_um, settle_s=None):
    """
    Go to `target_um` from ABOVE: first to target + overshoot, then down
    to the target, so the target is reached moving in the same (negative)
    direction as a descending scan, whatever the direction of the
    previous move - this takes up the mechanical backlash.
    """
    stage.move_to(axis, target_um + overshoot_um, settle_s)
    stage.move_to(axis, target_um, settle_s)


def scan(stage, axis, positions, overshoot_um, acquire, on_step=None):
    """
    Visit `positions` (strictly descending) along `axis` and call
    `acquire()` at each one. The stage first overshoots above the first
    position and comes back down to it, then only ever moves down.

    Returns (results, readback): `acquire()`'s return values and the
    position the stage reported at each plane.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.size > 1 and not np.all(np.diff(positions) < 0):
        raise ValueError("Scan positions must be strictly descending (one direction only).")

    approach(stage, axis, positions[0], overshoot_um)
    results, readback = [], []
    for i, target in enumerate(positions):
        if i:
            stage.move_to(axis, target)
        readback.append(stage.position(axis))
        results.append(acquire())
        if on_step is not None:
            on_step(i, target)
    return results, np.asarray(readback)
