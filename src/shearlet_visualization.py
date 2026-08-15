"""Visualization-only helpers for Gate 2 numeric scale-energy maps."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

COLORMAP = "cividis"
OUTSIDE_COLOR = "#f2f2f2"
MODE_LABELS = {1: "Mode 1 — parallel to y", 2: "Mode 2 — perpendicular to y"}


def _cmap():
    cmap = plt.get_cmap(COLORMAP).copy()
    cmap.set_bad(OUTSIDE_COLOR)
    return cmap


def _show(ax, axis1, axis2, values, mask, vmax):
    return ax.imshow(np.ma.array(values, mask=~mask), origin="lower",
                     extent=[axis1.min(), axis1.max(), axis2.min(), axis2.max()],
                     aspect="equal", interpolation="nearest", cmap=_cmap(), vmin=0.0, vmax=vmax)


def save_level_map(axis1, axis2, values, mask, out_path, title, vmax):
    if vmax is None or not np.isfinite(vmax) or vmax <= 0:
        raise ValueError("A positive global per-level vmax is required")
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.0), layout="constrained")
    image = _show(ax, np.asarray(axis1), np.asarray(axis2), np.asarray(values), np.asarray(mask), vmax)
    ax.set_title(title); ax.set_xlabel("x [mm]" if "T" in title.split(" · ")[0] else "y [mm]")
    ax.set_ylabel("z [mm]"); fig.colorbar(image, ax=ax, label="RMS-normalized Shearlet response")
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white"); plt.close(fig)


def save_representative_levels(records, out_path, vmax):
    if len(records) != 2:
        raise ValueError("Representative figure requires Mode1 and Mode2")
    records = sorted(records, key=lambda r: r["mode"])
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), squeeze=False)
    for row, record in enumerate(records):
        for column, scale in enumerate((1, 2, 3)):
            image = _show(axes[row, column], record["axis1"], record["axis2"],
                          record["maps"][scale], record["mask"], vmax[scale])
            ordering = {1: "coarser", 2: "intermediate", 3: "finer"}
            axes[row, column].set_title(
                f"{MODE_LABELS[record['mode']]} · Level {scale} ({ordering[scale]})"
            )
            axes[row, column].set_xlabel("x [mm]" if record["plane"] == "XZ" else "y [mm]")
            axes[row, column].set_ylabel("z [mm]")
            fig.colorbar(image, ax=axes[row, column], fraction=0.046, pad=0.04)
    fig.tight_layout(); out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600 if out_path.suffix == ".png" else None, bbox_inches="tight")
    plt.close(fig)


def save_fraction_comparison(rows, out_path):
    modes = (1, 2); scales = (1, 2, 3); rois = (
        ("scale_energy_fraction_full_phantom", "full phantom", "-"),
        ("scale_energy_fraction_interior", "interior 1.5 mm", "--"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True, layout="constrained")
    x = np.arange(3)
    colors = {1: "tab:blue", 2: "tab:orange"}
    for ax, plane in zip(axes, ("XZ", "YZ")):
        for mode in modes:
            for key, roi_label, linestyle in rois:
                samples = [[float(r[key]) for r in rows if r["plane"] == plane
                            and int(r["mode"]) == mode and int(r["scale"]) == scale]
                           for scale in scales]
                if any(len(values) != 10 for values in samples):
                    raise ValueError(f"Expected 10 {plane} cuts per mode/level/ROI")
                means = [np.mean(values) for values in samples]
                stds = [np.std(values, ddof=1) for values in samples]
                ax.errorbar(x, means, yerr=stds, marker="o", linestyle=linestyle,
                            color=colors[mode], capsize=3,
                            label=f"{MODE_LABELS[mode]} · {roi_label}")
        ax.set_title(plane); ax.set_xticks(x, ["Level 1\n(coarser)", "Level 2\n(intermediate)",
                                           "Level 3\n(finer)"])
        ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean Shearlet coefficient-energy fraction ± SD")
    axes[1].legend(fontsize=8)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600 if out_path.suffix == ".png" else None, bbox_inches="tight")
    plt.close(fig)
