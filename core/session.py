"""
core/session.py

Creates a uniquely-named run folder for each experiment:

    <root_folder>/YYYYMMDD/YYYYMMDD_<script_name>_<N>/

<script_name> is normally passed in explicitly by the experiment
script (cleanest and most reliable); N auto-increments with no
zero-padding, starting at 1.
"""
from datetime import datetime
from pathlib import Path


def create_run_folder(root_folder, script_name):
    root = Path(root_folder)
    date_str = datetime.now().strftime("%Y%m%d")
    day_folder = root / date_str
    day_folder.mkdir(parents=True, exist_ok=True)

    n = 1
    while True:
        candidate = day_folder / f"{date_str}_{script_name}_{n}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        n += 1


class Session:
    """Bundles a run folder with the loaded config for one experiment run."""

    def __init__(self, config, script_name, root_folder=None):
        self.config = config
        root = root_folder or config.get("acquisition_defaults", "root_folder", default="./data")
        self.run_folder = create_run_folder(root, script_name)

    @property
    def path(self):
        return self.run_folder
