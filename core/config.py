"""
core/config.py

Loads config/microscope_config.json and exposes it through a small
wrapper so the rest of the codebase never touches the raw dict (and
so a future switch in file format only touches this one module).
"""
import json
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "microscope_config.json"


class MicroscopeConfig:
    """Thin, read-only wrapper around microscope_config.json."""

    def __init__(self, config_path=DEFAULT_CONFIG_PATH):
        self.path = Path(config_path)
        with open(self.path, "r") as f:
            self._data = json.load(f)

    def __getitem__(self, key):
        return self._data[key]

    def get(self, *keys, default=None):
        """
        Safely walk a nested path of keys.

        Example: config.get("slm", "monitor_index")
        """
        node = self._data
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    @property
    def raw(self):
        """Full underlying dict, for cases that need it directly."""
        return self._data


def load_config(config_path=DEFAULT_CONFIG_PATH):
    return MicroscopeConfig(config_path)
