"""
core/mmcore.py

pymmcore-plus initialization: builds a configured CMMCorePlus instance
from the `pymmcore_plus` section of microscope_config.json (device
adapter search path + system configuration file).

This is the entry point for the new pymmcore-plus / napari-micromanager
stack. It doesn't touch the pycromanager-based code in functions/ and
experiments/, which remains as legacy for the existing MM-GUI-driven
workflow.
"""
from pymmcore_plus import CMMCorePlus


def load_mmcore(config):
    """
    Build a CMMCorePlus singleton configured from
    `pymmcore_plus.device_adapter_path` / `pymmcore_plus.system_config_path`
    in the loaded MicroscopeConfig.
    """
    device_adapter_path = config.get("pymmcore_plus", "device_adapter_path")
    system_config_path = config.get("pymmcore_plus", "system_config_path")

    mmc = CMMCorePlus.instance()
    if device_adapter_path:
        mmc.setDeviceAdapterSearchPaths([device_adapter_path])
    if system_config_path:
        mmc.loadSystemConfiguration(system_config_path)
    return mmc
