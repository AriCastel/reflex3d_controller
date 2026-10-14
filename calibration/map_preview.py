"""
calibration/map_preview.py

Builds the map preview figure. Returns the figure rather than saving
it, so core/file_io.py stays the only thing that writes to disk.
"""
import numpy as np


def build_map_figure(x_centers, y_centers, d2_map, result, config, channel):
    import matplotlib.pyplot as plt

    xs = np.asarray(x_centers, float)
    ys = np.asarray(y_centers, float)
    d2 = np.asarray(d2_map, float)

    dx = float(np.mean(np.diff(xs))) if len(xs) > 1 else 1.0
    dy = float(np.mean(np.diff(ys))) if len(ys) > 1 else 1.0
    extent = [xs[0] - dx / 2, xs[-1] + dx / 2, ys[-1] + dy / 2, ys[0] - dy / 2]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- left: the d^2 map with the estimate overlaid ---
    ax = axes[0]
    im = ax.imshow(d2, origin="upper", extent=extent, cmap="viridis",
                   interpolation="nearest", aspect="equal")
    fig.colorbar(im, ax=ax, label=r"$d^2$  (camera px$^2$)")

    cx, cy = result["center_x"], result["center_y"]
    if np.isfinite(cx) and np.isfinite(cy):
        ax.plot(cx, cy, "r+", markersize=18, markeredgewidth=2.5,
                label=f"estimate ({cx:.0f}, {cy:.0f})")
        r_pupil = config.get("fourier_plane", "radius_px")
        if r_pupil:
            ax.add_patch(plt.Circle((cx, cy), r_pupil, fill=False,
                                    color="red", ls="--", lw=1.5,
                                    label=f"pupil r={r_pupil} px"))
    gx, gy = result["centroid_center_x"], result["centroid_center_y"]
    if np.isfinite(gx) and result["method"] != "support_centroid":
        ax.plot(gx, gy, "wx", markersize=10, markeredgewidth=2,
                label="support centroid")

    width, height = config.get("slm", "resolution", default=[1920, 1080])
    ax.add_patch(plt.Rectangle((0, 0), width, height, fill=False,
                               color="white", lw=1.0))
    ax.set_xlabel("SLM x (px)")
    ax.set_ylabel("SLM y (px)")
    ax.set_title(f"{result['mode']} probe — expect a "
                 f"{result['expected_extremum']} at the centre")
    ax.legend(loc="upper right", fontsize=8)

    # --- right: radial profile about the estimate, as a sanity check ---
    ax2 = axes[1]
    XG, YG = np.meshgrid(xs, ys)
    if np.isfinite(cx) and np.isfinite(cy):
        r = np.sqrt((XG - cx) ** 2 + (YG - cy) ** 2)
        ok = np.isfinite(d2)
        ax2.plot(r[ok], d2[ok], ".", ms=4, alpha=0.5, color="0.5",
                 label="map points")
        # binned mean profile
        nb = 15
        edges = np.linspace(0, np.nanmax(r[ok]), nb + 1)
        cen, prof = [], []
        for i in range(nb):
            sel = ok & (r >= edges[i]) & (r < edges[i + 1])
            if sel.sum():
                cen.append(0.5 * (edges[i] + edges[i + 1]))
                prof.append(np.nanmean(d2[sel]))
        ax2.plot(cen, prof, "-o", color="crimson", ms=4,
                 label="binned mean")
        fit_r = result.get("fit_radius_px")
        if fit_r and np.isfinite(fit_r):
            ax2.axvline(fit_r, color="steelblue", ls="--",
                        label=f"fit region edge ({fit_r:.0f} px)")
        thr = result.get("support_threshold")
        if thr and np.isfinite(thr):
            ax2.axhline(thr, color="green", ls=":", label="support threshold")
    ax2.set_xlabel("distance from estimated centre (SLM px)")
    ax2.set_ylabel(r"$d^2$  (camera px$^2$)")
    ax2.set_title("Radial profile")
    ax2.legend(fontsize=8)

    fig.suptitle(
        f"Fourier-plane alignment — channel '{channel}'   "
        f"[{result['method']}, {result['n_failed_points']} failed pts]",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def build_refinement_figure(x_centers, y_centers, d2_map, result, config,
                            channel):
    """
    Preview of a fine refinement raster: where on the SLM it was taken,
    the fine d^2 map with the previous and refined centres, and the
    radial profile about the refined centre.
    """
    import matplotlib.pyplot as plt

    xs = np.asarray(x_centers, float)
    ys = np.asarray(y_centers, float)
    d2 = np.asarray(d2_map, float)
    px, py = result["previous_center_x"], result["previous_center_y"]
    cx, cy = result["center_x"], result["center_y"]
    r_win = result["window_radius_px"]
    r_pupil = config.get("fourier_plane", "radius_px")
    width, height = config.get("slm", "resolution", default=[1920, 1080])
    have_new = np.isfinite(cx) and np.isfinite(cy)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5),
                             gridspec_kw={"width_ratios": [1.0, 1.25, 1.0]})

    # --- left: the rastered area on the whole SLM ---
    ax = axes[0]
    ax.add_patch(plt.Rectangle((0, 0), width, height, fill=False,
                               color="black", lw=1.2))
    ax.add_patch(plt.Circle((px, py), r_win, color="tab:blue", alpha=0.35,
                            label=f"rastered area (r={r_win:.0f} px)"))
    if r_pupil:
        ax.add_patch(plt.Circle((px, py), r_pupil, fill=False, color="red",
                                ls="--", lw=1.2, label=f"pupil r={r_pupil} px"))
    ax.plot(px, py, "wx", mec="k", markersize=9, label="previous centre")
    if have_new:
        ax.plot(cx, cy, "r+", markersize=14, markeredgewidth=2,
                label="refined centre")
    ax.set_xlim(-0.03 * width, 1.03 * width)
    ax.set_ylim(1.03 * height, -0.03 * height)
    ax.set_aspect("equal")
    ax.set_xlabel("SLM x (px)")
    ax.set_ylabel("SLM y (px)")
    ax.set_title("Rastered area on the SLM")
    ax.legend(loc="lower right", fontsize=7)

    # --- middle: the fine map ---
    ax = axes[1]
    dx = float(np.mean(np.diff(xs))) if len(xs) > 1 else 1.0
    dy = float(np.mean(np.diff(ys))) if len(ys) > 1 else 1.0
    extent = [xs[0] - dx / 2, xs[-1] + dx / 2, ys[-1] + dy / 2, ys[0] - dy / 2]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.9")
    im = ax.imshow(np.ma.masked_invalid(d2), origin="upper", extent=extent,
                   cmap=cmap, interpolation="nearest", aspect="equal")
    fig.colorbar(im, ax=ax, label=r"$d^2$  (camera px$^2$)")
    ax.add_patch(plt.Circle((px, py), r_win, fill=False, color="white",
                            ls=":", lw=1.2))
    ax.plot(px, py, "wx", markersize=10, markeredgewidth=2,
            label=f"previous ({px:.1f}, {py:.1f})")
    if have_new:
        ax.plot(cx, cy, "r+", markersize=18, markeredgewidth=2.5,
                label=f"refined ({cx:.1f}, {cy:.1f})")
    ax.set_xlabel("SLM x (px)")
    ax.set_ylabel("SLM y (px)")
    ax.set_title(f"{result['mode']} probe, step {dx:.0f} px — expect a "
                 f"{result['expected_extremum']} at the centre", fontsize=10)
    ax.legend(loc="best", fontsize=7)

    # --- right: radial profile about the refined centre ---
    ax = axes[2]
    if have_new:
        XG, YG = np.meshgrid(xs, ys)
        r = np.hypot(XG - cx, YG - cy)
        ok = np.isfinite(d2)
        ax.plot(r[ok], d2[ok], ".", ms=3, alpha=0.4, color="0.5",
                label="map points")
        nb = 15
        edges = np.linspace(0, r[ok].max(), nb + 1)
        cen, prof = [], []
        for i in range(nb):
            sel = ok & (r >= edges[i]) & (r < edges[i + 1])
            if sel.sum():
                cen.append(0.5 * (edges[i] + edges[i + 1]))
                prof.append(np.nanmean(d2[sel]))
        ax.plot(cen, prof, "-o", color="crimson", ms=4, label="binned mean")
    ax.set_xlabel("distance from refined centre (SLM px)")
    ax.set_ylabel(r"$d^2$  (camera px$^2$)")
    ax.set_title("Radial profile")
    ax.legend(fontsize=8)

    fig.suptitle(
        f"Fourier-plane refinement — channel '{channel}'   "
        f"[shift {result['shift_px']:.1f} px, radial model explains "
        f"{100 * result['explained_fraction']:.0f}% of variance, "
        f"{result['n_failed_points']} failed pts]",
        fontsize=11,
    )
    fig.tight_layout()
    return fig
