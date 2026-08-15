from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import numpy as np

from .dataset import (
    EXPECTED_XZ, EXPECTED_YZ, POLARIZATION_BY_MODE, FieldPair,
    discover_phantom_pairs, validate_expected_40,
)
from .fld_reader import pair_real_imag
from .field_processing import field_magnitude, reshape_plane
from .geometry_masks import mask_xz, mask_yz, apply_mask
from .visualization import save_intensity_image

EXPECTED_OUTPUTS = 40
PRESENTATION_PERCENTILE = 99.5
FREQUENCY_GHZ = 2.45
PHANTOM_RADIUS_MM = 50.0
PHANTOM_LENGTH_MM = 200.0
POLARIZATION_FIGURE_LABEL = {
    1: "parallel to y",
    2: "perpendicular to y",
}


@dataclass(frozen=True)
class _PreparedCut:
    pair: FieldPair
    axis1: np.ndarray
    axis2: np.ndarray
    raw: np.ndarray
    mask: np.ndarray
    masked: np.ndarray
    cut_name: str
    xlabel: str
    ylabel: str


def _validate_cut_coordinates(pair: FieldPair, coords: np.ndarray, coord_unit: str) -> None:
    if coord_unit.strip().lower() != "mm":
        raise ValueError(
            f"{pair.real_path.name}: expected coordinate unit mm, found {coord_unit!r}"
        )
    fixed_axis = 1 if pair.plane == "XZ" else 0
    fixed_values = coords[:, fixed_axis]
    if not np.allclose(fixed_values, pair.fixed_mm, rtol=0, atol=1e-9):
        name = "y" if pair.plane == "XZ" else "x"
        raise ValueError(
            f"{pair.real_path.name}: expected fixed {name}={pair.fixed_mm} mm, "
            f"found range [{fixed_values.min()}, {fixed_values.max()}] mm"
        )


def _clear_generated_files(output_root: Path) -> None:
    """Remove only artifacts owned by this pipeline so reruns have exact counts."""
    for relative, suffix in (("01_intensity", ".png"), ("02_numeric", ".npz")):
        directory = output_root / relative
        if directory.exists():
            for path in directory.rglob(f"*{suffix}"):
                path.unlink()


def run_intensity_pipeline(
    data_root: str | Path,
    output_root: str | Path,
    *,
    dataset_filename: str | None = None,
    dataset_sha256: str | None = None,
    git_commit_sha: str | None = None,
):
    data_root = Path(data_root)
    output_root = Path(output_root)
    pairs = discover_phantom_pairs(data_root)
    validate_expected_40(pairs)

    # Validate and reconstruct every scientific input before writing artifacts.
    prepared: list[_PreparedCut] = []
    for pair in pairs:
        coords, electric_field, meta = pair_real_imag(pair.real_path, pair.imag_path)
        _validate_cut_coordinates(pair, coords, meta.coord_unit)
        magnitude = field_magnitude(electric_field)
        axis1, axis2, raw = reshape_plane(coords, magnitude, pair.plane)

        if pair.plane == "XZ":
            mask = mask_xz(axis1, axis2)
            xlabel, ylabel = "x [mm]", "z [mm]"
            cut_name = f"T{pair.index:02d}"
        else:
            mask = mask_yz(axis1, axis2, pair.fixed_mm)
            xlabel, ylabel = "y [mm]", "z [mm]"
            cut_name = f"A{pair.index:02d}"

        if not mask.any():
            raise ValueError(f"{pair.real_path.name}: no samples lie inside the phantom")
        masked = apply_mask(raw, mask, outside_value=0.0)
        prepared.append(_PreparedCut(
            pair, axis1, axis2, raw, mask, masked, cut_name, xlabel, ylabel
        ))

    if len(prepared) != EXPECTED_OUTPUTS:
        raise RuntimeError(f"Expected {EXPECTED_OUTPUTS} prepared cuts, got {len(prepared)}")

    # Both products use one scale across all 40 cuts; matrices remain untouched.
    all_inside = np.concatenate([cut.raw[cut.mask] for cut in prepared])
    physical_vmin = 0.0
    physical_vmax = float(np.max(all_inside))
    presentation_vmax = float(np.percentile(all_inside, PRESENTATION_PERCENTILE))
    if physical_vmax <= physical_vmin:
        raise ValueError("All in-phantom field magnitudes are zero; cannot plot a scale")
    if presentation_vmax <= physical_vmin:
        raise ValueError("Global robust electric-field magnitude limit is not positive")

    _clear_generated_files(output_root)
    rows = []
    for cut in prepared:
        pair = cut.pair
        physical_png_path = (
            output_root / "01_intensity" / "physical_shared" / f"Mode{pair.mode}" /
            pair.plane / f"{cut.cut_name}.png"
        )
        presentation_png_path = (
            output_root / "01_intensity" / "presentation_shared" / f"Mode{pair.mode}" /
            pair.plane / f"{cut.cut_name}.png"
        )
        matrix_path = (
            output_root / "02_numeric" / f"Mode{pair.mode}" /
            pair.plane / f"{cut.cut_name}.npz"
        )
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            matrix_path,
            intensity_V_per_m=cut.masked,
            unmasked_intensity_V_per_m=cut.raw,
            inside_solid=cut.mask,
            axis1_mm=cut.axis1,
            axis2_mm=cut.axis2,
            plane=pair.plane,
            fixed_mm=pair.fixed_mm,
            mode=pair.mode,
            polarization=POLARIZATION_BY_MODE[pair.mode],
        )

        polarization = POLARIZATION_BY_MODE[pair.mode]
        fixed_axis = "y" if pair.plane == "XZ" else "x"
        title_line_1 = (
            f"{cut.cut_name} · {pair.plane} · {fixed_axis} = {pair.fixed_mm:+d} mm · "
            f"Mode {pair.mode} ({POLARIZATION_FIGURE_LABEL[pair.mode]})"
        )
        save_intensity_image(
            cut.axis1, cut.axis2, cut.masked, physical_png_path,
            f"{title_line_1}\nElectric-field magnitude |E| · {FREQUENCY_GHZ:g} GHz · physical shared scale",
            cut.xlabel, cut.ylabel,
            vmin=physical_vmin, vmax=physical_vmax,
        )
        save_intensity_image(
            cut.axis1, cut.axis2, cut.masked, presentation_png_path,
            f"{title_line_1}\nElectric-field magnitude |E| · {FREQUENCY_GHZ:g} GHz · presentation shared scale",
            cut.xlabel, cut.ylabel, vmin=physical_vmin, vmax=presentation_vmax,
        )

        inside = cut.raw[cut.mask]
        rows.append({
            "mode": pair.mode,
            "polarization": polarization,
            "plane": pair.plane,
            "cut": cut.cut_name,
            "fixed_mm": pair.fixed_mm,
            "samples_total": int(cut.raw.size),
            "samples_inside_solid": int(cut.mask.sum()),
            "E_mean_V_per_m": float(np.mean(inside)),
            "E_max_V_per_m": float(np.max(inside)),
            "E_rms_V_per_m": float(np.sqrt(np.mean(inside**2))),
            "physical_shared_png": str(physical_png_path.relative_to(output_root)),
            "presentation_shared_png": str(presentation_png_path.relative_to(output_root)),
            "numeric_matrix": str(matrix_path.relative_to(output_root)),
        })

    metrics_path = output_root / "04_metrics" / "field_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    run_metadata = {
        "dataset_filename": dataset_filename,
        "dataset_sha256": dataset_sha256,
        "git_commit_sha": git_commit_sha,
        "frequency_GHz": FREQUENCY_GHZ,
        "quantity": "Electric-field magnitude |E|",
        "unit": "V/m",
        "phantom_radius_mm": PHANTOM_RADIUS_MM,
        "phantom_length_mm": PHANTOM_LENGTH_MM,
        "XZ_cut_positions_mm": list(EXPECTED_XZ),
        "YZ_cut_positions_mm": list(EXPECTED_YZ),
        "cut_count": len(prepared),
        "physical_shared_vmin_V_per_m": physical_vmin,
        "physical_shared_vmax_V_per_m": physical_vmax,
        "presentation_percentile": PRESENTATION_PERCENTILE,
        "presentation_shared_vmin_V_per_m": physical_vmin,
        "presentation_shared_vmax_V_per_m": presentation_vmax,
        "polarization_by_mode": {f"Mode{mode}": value for mode, value in POLARIZATION_BY_MODE.items()},
    }
    metadata_path = output_root / "04_metrics" / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(run_metadata, stream, indent=2)
        stream.write("\n")

    physical_pngs = list((output_root / "01_intensity" / "physical_shared").rglob("*.png"))
    presentation_pngs = list((output_root / "01_intensity" / "presentation_shared").rglob("*.png"))
    matrices = list((output_root / "02_numeric").rglob("*.npz"))
    if not all(len(items) == EXPECTED_OUTPUTS for items in (physical_pngs, presentation_pngs, matrices, rows)):
        raise RuntimeError(
            f"Gate 1 output count mismatch: {len(physical_pngs)} physical PNGs, "
            f"{len(presentation_pngs)} presentation PNGs, {len(matrices)} matrices, "
            f"{len(rows)} metric rows"
        )
    return rows
