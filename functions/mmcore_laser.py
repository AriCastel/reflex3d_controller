"""
functions/mmcore_laser.py

pymmcore-plus equivalent of functions/laser.py: same laser on/off/power
control, addressed through a CMMCorePlus instance's native (camelCase)
API instead of pycromanager's Java-proxy. Never writes to disk.
"""


def set_laser_power_mmcore(mmc, config, power_mW):
    label = config.get("laser", "device_label")
    prop = config.get("laser", "power_property", default="12-Power Setpoint [mW]")
    mmc.setProperty(label, prop, power_mW)


def laser_on_mmcore(mmc, config):
    label = config.get("laser", "device_label")
    prop = config.get("laser", "state_property", default="10-Emission Status")
    mmc.setProperty(label, prop, "open")


def laser_off_mmcore(mmc, config):
    label = config.get("laser", "device_label")
    prop = config.get("laser", "state_property", default="10-Emission Status")
    mmc.setProperty(label, prop, "closed")
