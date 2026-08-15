
from __future__ import annotations
from pathlib import Path
import csv
import numpy as np

from .dataset import discover_phantom_pairs, validate_expected_40
from .fld_reader import pair_real_imag
from .field_processing import field_magnitude, reshape_plane
from .geometry_masks import mask_xz, mask_yz, apply_mask
from .visualization import save_intensity_image

def run_intensity_pipeline(data_root: str | Path, output_root: str | Path):
    data_root = Path(data_root)
    output_root = Path(output_root)
    pairs = discover_phantom_pairs(data_root)
    validate_expected_40(pairs)

    rows = []
    for p in pairs:
        coords, E, meta = pair_real_imag(p.real_path, p.imag_path)
        mag = field_magnitude(E)
        axis1, axis2, img = reshape_plane(coords, mag, p.plane)

        if p.plane == "XZ":
            mask = mask_xz(axis1, axis2)
            xlabel, ylabel = "x [mm]", "z [mm]"
            cut_name = f"T{p.index:02d}"
        else:
            mask = mask_yz(axis1, axis2, p.fixed_mm)
            xlabel, ylabel = "y [mm]", "z [mm]"
            cut_name = f"A{p.index:02d}"

        masked = apply_mask(img, mask, outside_value=0.0)
        out = output_root / "01_intensity" / f"Mode{p.mode}" / p.plane / f"{cut_name}.png"
        title = f"{cut_name} | {p.plane} | fixed={p.fixed_mm:+d} mm | Mode {p.mode} | 2.45 GHz"
        save_intensity_image(axis1, axis2, masked, out, title, xlabel, ylabel)

        inside = img[mask]
        rows.append({
            "mode": p.mode,
            "plane": p.plane,
            "cut": cut_name,
            "fixed_mm": p.fixed_mm,
            "samples_total": int(img.size),
            "samples_inside_solid": int(mask.sum()),
            "E_mean_V_per_m": float(np.mean(inside)),
            "E_max_V_per_m": float(np.max(inside)),
            "E_rms_V_per_m": float(np.sqrt(np.mean(inside**2))),
            "output_png": str(out.relative_to(output_root)),
        })

    metrics = output_root / "04_metrics" / "field_metrics.csv"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    with metrics.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    return rows
