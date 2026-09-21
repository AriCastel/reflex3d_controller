"""
functions/laser.py (legacy, pycromanager)

General-purpose laser control via the Micro-Manager core, addressed
through pycromanager's `Core` (whose Java-proxy translates these
snake_case names to `setProperty` etc.). Never writes to disk —
that's core/file_io.py's job.

Kept for the existing MM-GUI-driven workflow. functions/mmcore_laser.py
has the pymmcore-plus equivalent, used by the `*_mda` experiments.
"""


def set_laser_power(core, config, power_mW):
    label = config.get("laser", "device_label")
    prop = config.get("laser", "power_property", default="12-Power Setpoint [mW]")
    core.set_property(label, prop, power_mW)


def laser_on(core, config):
    label = config.get("laser", "device_label")
    prop = config.get("laser", "state_property", default="10-Emission Status")
    core.set_property(label, prop, 'open')


def laser_off(core, config):
    label = config.get("laser", "device_label")
    prop = config.get("laser", "state_property", default="10-Emission Status")
    core.set_property(label, prop, 'closed')
