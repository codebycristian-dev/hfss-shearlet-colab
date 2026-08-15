
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ARTICLE_COLORMAP = "cividis"
ARTICLE_OUTSIDE_MASK_COLOR = "#f2f2f2"
COLORBAR_LABEL = r"Electric-field magnitude $|E|$ [V/m]"

def save_intensity_image(axis1, axis2, intensity, out_path, title, xlabel, ylabel,
                         cmap="gray", vmin=None, vmax=None):
    """
    Electric-field magnitude visualization only. The scientific matrix remains single-channel float.
    No RGB array is used as input to later transforms.
    """
    intensity = np.asarray(intensity)
    if intensity.ndim != 2 or not np.issubdtype(intensity.dtype, np.number):
        raise ValueError("Electric-field magnitude visualization input must be a numeric 2D matrix")
    if not np.isfinite(intensity).all():
        raise ValueError("Electric-field magnitude visualization input contains NaN/Inf")
    if vmin is None or vmax is None:
        raise ValueError("Explicit shared vmin/vmax are required; per-image normalization is forbidden")
    if not np.isfinite([vmin, vmax]).all() or vmax <= vmin:
        raise ValueError(f"Invalid visualization limits: vmin={vmin}, vmax={vmax}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 5.0), layout="constrained")
    im = ax.imshow(
        intensity,
        origin="lower",
        extent=[axis1.min(), axis1.max(), axis2.min(), axis2.max()],
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=10, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, pad=0.04, label=COLORBAR_LABEL)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _article_colormap(outside_mask_color=ARTICLE_OUTSIDE_MASK_COLOR):
    cmap = plt.get_cmap(ARTICLE_COLORMAP).copy()
    cmap.set_bad(outside_mask_color)
    return cmap


def _validate_article_arrays(axis1, axis2, intensity, mask, vmin, vmax):
    axis1 = np.asarray(axis1)
    axis2 = np.asarray(axis2)
    intensity = np.asarray(intensity)
    mask = np.asarray(mask)
    if intensity.ndim != 2 or not np.issubdtype(intensity.dtype, np.number):
        raise ValueError("Article visualization input must be a numeric 2D matrix")
    if mask.shape != intensity.shape or mask.dtype.kind != "b":
        raise ValueError("Article visualization mask must be boolean and match the matrix shape")
    if not np.isfinite(intensity).all():
        raise ValueError("Article visualization input contains NaN/Inf")
    if vmin is None or vmax is None:
        raise ValueError("Explicit shared vmin/vmax are required; per-image normalization is forbidden")
    if not np.isfinite([vmin, vmax]).all() or vmax <= vmin:
        raise ValueError(f"Invalid visualization limits: vmin={vmin}, vmax={vmax}")
    return axis1, axis2, intensity, mask


def _imshow_article(ax, axis1, axis2, intensity, mask, *, vmin, vmax, cmap):
    return ax.imshow(
        np.ma.array(intensity, mask=~mask),
        origin="lower",
        extent=[axis1.min(), axis1.max(), axis2.min(), axis2.max()],
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )


def save_article_image(axis1, axis2, intensity, mask, out_path, title, xlabel, ylabel,
                       *, vmin, vmax, outside_mask_color=ARTICLE_OUTSIDE_MASK_COLOR):
    """Render a publication figure without altering its numeric scientific matrix."""
    axis1, axis2, intensity, mask = _validate_article_arrays(
        axis1, axis2, intensity, mask, vmin, vmax
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.0), layout="constrained")
    image = _imshow_article(
        ax, axis1, axis2, intensity, mask,
        vmin=vmin, vmax=vmax, cmap=_article_colormap(outside_mask_color),
    )
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(image, ax=ax, pad=0.04, label=COLORBAR_LABEL)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_article_mosaic(panels, out_path, *, nrows, ncols, vmin, vmax,
                        outside_mask_color=ARTICLE_OUTSIDE_MASK_COLOR):
    """Render article panels with one global scale and one shared colorbar."""
    if len(panels) != nrows * ncols:
        raise ValueError(f"Expected {nrows * ncols} mosaic panels, got {len(panels)}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width = 24 if ncols == 10 else 10
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, 6.5), squeeze=False)
    cmap = _article_colormap(outside_mask_color)
    image = None
    for ax, panel in zip(axes.flat, panels):
        axis1, axis2, intensity, mask = _validate_article_arrays(
            panel["axis1"], panel["axis2"], panel["intensity"], panel["mask"], vmin, vmax
        )
        image = _imshow_article(
            ax, axis1, axis2, intensity, mask, vmin=vmin, vmax=vmax, cmap=cmap
        )
        ax.set_title(panel["title"], fontsize=8, pad=4)
        ax.set_xlabel(panel["xlabel"], fontsize=7)
        ax.set_ylabel(panel["ylabel"], fontsize=7)
        ax.tick_params(labelsize=6)
    fig.subplots_adjust(left=0.05, right=0.91, bottom=0.09, top=0.93, wspace=0.38, hspace=0.32)
    colorbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
    fig.colorbar(image, cax=colorbar_ax, label=COLORBAR_LABEL)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
