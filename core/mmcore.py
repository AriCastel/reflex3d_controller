"""
core/mmcore.py

pymmcore-plus initialization: builds a configured CMMCorePlus instance
from the `pymmcore_plus` section of microscope_config.json (device
adapter search path, system configuration file, sequence buffer size).
"""
from pymmcore_plus import CMMCorePlus


def load_mmcore(config):
    """The CMMCorePlus singleton, configured from the `pymmcore_plus` config section."""
    device_adapter_path = config.get("pymmcore_plus", "device_adapter_path")
    system_config_path = config.get("pymmcore_plus", "system_config_path")

    mmc = CMMCorePlus.instance()
    if device_adapter_path:
        mmc.setDeviceAdapterSearchPaths([device_adapter_path])
    if system_config_path:
        mmc.loadSystemConfiguration(system_config_path)
    # Micro-Manager's own RAM buffer that a camera sequence fills before the
    # frames are collected (the GUI's Tools > Options > "Sequence buffer size").
    # Larger = more slack for very fast acquisitions; None keeps the default.
    buffer_mb = config.get("pymmcore_plus", "sequence_buffer_mb")
    if buffer_mb:
        mmc.setCircularBufferMemoryFootprint(int(buffer_mb))

    device_labels = mmc.getLoadedDevices()

    for label in device_labels:
        print(label)

    return mmc
