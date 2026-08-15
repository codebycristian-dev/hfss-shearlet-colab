
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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
    fig.colorbar(im, ax=ax, pad=0.04, label=r"Electric-field magnitude $|E|$ [V/m]")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
