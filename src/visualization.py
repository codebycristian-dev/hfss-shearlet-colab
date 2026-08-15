
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def save_intensity_image(axis1, axis2, intensity, out_path, title, xlabel, ylabel,
                         cmap="gray", vmin=None, vmax=None):
    """
    Visualization only. The scientific matrix remains single-channel float.
    No RGB array is used as input to later transforms.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    im = ax.imshow(
        intensity,
        origin="lower",
        extent=[axis1.min(), axis1.max(), axis2.min(), axis2.max()],
        aspect="equal" if abs((axis1.max()-axis1.min()) - (axis2.max()-axis2.min())) < 60 else "auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, label=r"$|E|$ [V/m]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
