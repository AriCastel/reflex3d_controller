"""
functions/mmcore_rotator.py

The PR50CC rotation stage the sample is mounted on, addressed through a
CMMCorePlus instance. Never writes to disk.

The device has no "move by" or speed command: it is homed by setting its
home property to a command string, and moved by writing its absolute
position - the 'X Axis' property, in milli-degrees (90000 = 90 degrees
clockwise, -90000 = 90 degrees counter-clockwise). A relative rotation is
therefore done here by reading the current position and writing it back
plus the change. Everything outside this module works in degrees,
positive clockwise; rotator.* in the config holds the device specifics.
"""
import time


class Rotator:
    """
    Usage:
        rotator = Rotator(mmc, config)
        rotator.initialize()          # find home - once after power-up
        rotator.rotate_by(-3.0)       # 3 degrees counter-clockwise
    """

    def __init__(self, mmc, config):
        self.mmc = mmc
        cfg = config.get("rotator", default={})
        self.label = cfg.get("device_label", "PR50CC")
        self.position_property = cfg.get("position_property", "X Axis")
        self.home_property = cfg.get("home_property", "find_home")
        self.home_command = cfg.get("home_command", "FIND HOME NOW")
        self.units_per_degree = float(cfg.get("units_per_degree", 1000))
        self.min_deg = cfg.get("min_deg")
        self.max_deg = cfg.get("max_deg")
        self.settle_s = float(cfg.get("settle_s", 0.3))
        self.coarse_step_deg = float(cfg.get("coarse_step_deg", 45.0))
        self.fine_step_deg = float(cfg.get("fine_step_deg", 5.0))
        if self.units_per_degree <= 0:
            raise ValueError("rotator.units_per_degree must be positive.")

    def _wait(self):
        self.mmc.waitForDevice(self.label)
        time.sleep(self.settle_s)

    def initialize(self):
        """Find the home position (the stage rotates there; blocks until it's done)."""
        self.mmc.setProperty(self.label, self.home_property, self.home_command)
        self._wait()
        return self.position_deg()

    def position_deg(self):
        """Current angle in degrees, as the stage reports it."""
        raw = float(self.mmc.getProperty(self.label, self.position_property))
        return raw / self.units_per_degree

    def move_to(self, angle_deg):
        """Rotate to an absolute angle (degrees); returns the angle read back."""
        if self.min_deg is not None and angle_deg < self.min_deg or \
                self.max_deg is not None and angle_deg > self.max_deg:
            raise ValueError(
                f"{angle_deg:.3f} deg is outside rotator.min_deg/max_deg "
                f"[{self.min_deg}, {self.max_deg}]."
            )
        units = int(round(angle_deg * self.units_per_degree))
        self.mmc.setProperty(self.label, self.position_property, str(units))
        self._wait()
        return self.position_deg()

    def rotate_by(self, delta_deg):
        """
        Rotate by `delta_deg` (positive = clockwise): the current position
        read from the stage plus the change, written back as the new
        absolute position. Returns the angle read back.
        """
        units = round(self.position_deg() * self.units_per_degree)
        return self.move_to((units + round(delta_deg * self.units_per_degree))
                            / self.units_per_degree)
