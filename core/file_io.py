"""
core/file_io.py

The sole authority for all disk writes in this codebase. Nothing in
functions/ or experiments/ should touch the filesystem directly —
every save (TIFFs, metadata, anything added later) is routed through
here, so the output format stays consistent in one place.
"""
import json
from pathlib import Path

import numpy as np
import tifffile


def save_timelapse_tiff(run_folder, frames, metadata, filename="timelapse.tif"):
    """
    Save a timelapse as an ImageJ-compatible hyperstack with TZCYX axis
    ordering (kept even for this single-Z/single-channel case, for
    forward compatibility with planned 4D T+Z acquisitions).

    Parameters
    ----------
    run_folder : Path or str
    frames : np.ndarray
        Shape (T, Y, X) for this single-channel, single-plane use case.
        Reshaped internally to (T, Z=1, C=1, Y, X).
    metadata : dict
        Per-run metadata to embed (exposure, laser power, SLM mask note,
        per-frame timestamps, etc.).
    filename : str

    Returns
    -------
    Path to the written TIFF.
    """
    run_folder = Path(run_folder)
    stack = np.asarray(frames)

    if stack.ndim == 3:
        t, y, x = stack.shape
        stack = stack.reshape(t, 1, 1, y, x)  # -> TZCYX
    elif stack.ndim != 5:
        raise ValueError(f"Expected a (T,Y,X) or (T,Z,C,Y,X) array, got shape {stack.shape}")

    out_path = run_folder / filename
    tifffile.imwrite(
        out_path,
        stack,
        imagej=True,
        metadata={"axes": "TZCYX"},
    )

    # Full metadata (including things ImageJ tags can't hold well, like
    # per-frame timestamp lists) is also kept as a sidecar JSON.
    meta_path = run_folder / f"{Path(filename).stem}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return out_path


def save_frame_tiff(run_folder, frame, metadata=None, filename="frame.tif"):
    """Save a single 2D frame (e.g. the point-source verification snap)."""
    run_folder = Path(run_folder)
    out_path = run_folder / filename
    tifffile.imwrite(out_path, np.asarray(frame))
    if metadata is not None:
        meta_path = run_folder / f"{Path(filename).stem}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
    return out_path


def save_alignment_map(run_folder, x_centers, y_centers, d2_map, metadata,
                       basename="fourier_map"):
    """
    Save a Fourier-alignment d^2 map.

    Writes three things, deliberately redundantly: an .npz for exact
    round-tripping and re-analysis, a .csv for quick inspection in any
    other tool, and a metadata .json.
    """
    run_folder = Path(run_folder)

    npz_path = run_folder / f"{basename}.npz"
    np.savez(
        npz_path,
        x_centers=np.asarray(x_centers),
        y_centers=np.asarray(y_centers),
        d2_map=np.asarray(d2_map),
    )

    csv_path = run_folder / f"{basename}.csv"
    d2 = np.asarray(d2_map)
    with open(csv_path, "w") as f:
        f.write("slm_x_px,slm_y_px,d_squared_camera_px2\n")
        for iy, y in enumerate(y_centers):
            for ix, x in enumerate(x_centers):
                f.write(f"{x:.2f},{y:.2f},{d2[iy, ix]:.6f}\n")

    meta_path = run_folder / f"{basename}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return npz_path


def save_figure(run_folder, fig, filename="fourier_map_preview.png", dpi=150):
    """Save a matplotlib figure into the run folder."""
    run_folder = Path(run_folder)
    out_path = run_folder / filename
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return out_path


def write_fourier_center(config_path, channel, center_xy, extra=None):
    """
    Write a measured Fourier-plane centre into the JSON config, under
    fourier_plane.channels.<channel>.center_px, preserving everything
    else in the file.

    A backup of the previous config is written alongside it before the
    update, since this rewrites a file the whole codebase depends on.

    Returns (config_path, backup_path).
    """
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        data = json.load(f)

    backup_path = config_path.with_suffix(".json.bak")
    with open(backup_path, "w") as f:
        json.dump(data, f, indent=2)

    fourier = data.setdefault("fourier_plane", {})
    channels = fourier.setdefault("channels", {})
    entry = channels.setdefault(channel, {})
    entry["center_px"] = [float(center_xy[0]), float(center_xy[1])]
    if extra:
        entry.update(extra)

    with open(config_path, "w") as f:
        json.dump(data, f, indent=2)

    return config_path, backup_path
