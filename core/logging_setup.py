"""
core/logging_setup.py

One log file per run folder, plus a console echo so progress is
visible while an experiment is running.
"""
import logging
from pathlib import Path


def setup_logger(run_folder, name="reflex3d"):
    run_folder = Path(run_folder)
    log_path = run_folder / "run.log"

    logger = logging.getLogger(f"{name}.{run_folder.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(console_handler)

    return logger
