"""Gate 2: reproducible three-scale 2-D discrete Shearlet analysis."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
import csv
import json
import platform
import subprocess

import numpy as np
import scipy
import matplotlib
from scipy.ndimage import distance_transform_edt

try:
    import pyshearlab_mind as pyshearlab
except ImportError as exc:  # pragma: no cover - exercised in dependency checks
    raise ImportError(
        "Gate 2 requires the pinned backend pyShearLab-MIND==0.0.3. "
        "Install requirements.txt; no fallback transform is permitted."
    ) from exc

from .shearlet_visualization import (
    save_fraction_comparison,
    save_level_map,
    save_representative_levels,
)

BACKEND_DISTRIBUTION = "pyShearLab-MIND"
BACKEND_IMPORT = "pyshearlab_mind"
BACKEND_VERSION = "0.0.3"
USE_GPU = 0
N_SCALES = 3
SHEAR_LEVELS = (1, 1, 2)
FULL = 0
LOWPASS_INDEX = (0, 0, 0)
ROI_DISTANCE_MM = 1.5
VISUALIZATION_PERCENTILE = 99.5
EXPECTED_CUTS = 40
EXPECTED_LEVEL_MAPS = EXPECTED_CUTS * N_SCALES
RECONSTRUCTION_TOLERANCE = 1e-10
REPO_REF = "main"
EXPECTED_TOTAL_FILTERS = 33
EXPECTED_LOWPASS_FILTERS = 1
EXPECTED_FILTERS_PER_SCALE = {1: 8, 2: 8, 3: 16}
SCALE_INTERPRETATION = {
    1: "coarser spatial-scale band",
    2: "intermediate spatial-scale band",
    3: "finer spatial-scale band",
}


@dataclass(frozen=True)
class Padding:
    original_shape: tuple[int, int]
    padded_shape: tuple[int, int]
    pad_width: tuple[tuple[int, int], tuple[int, int]]
    mode: str = "reflect"


def backend_version() -> str:
    found = version(BACKEND_DISTRIBUTION)
    if found != BACKEND_VERSION:
        raise RuntimeError(
            f"Gate 2 requires {BACKEND_DISTRIBUTION}=={BACKEND_VERSION}, found {found}"
        )
    return found


def pad_to_power_of_two_square(matrix: np.ndarray) -> tuple[np.ndarray, Padding]:
    """Reflect-pad, without interpolation, to the smallest enclosing 2^n square."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or not np.issubdtype(matrix.dtype, np.number):
        raise ValueError("Shearlet input must be a numeric 2-D matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("Shearlet input contains NaN/Inf")
    side = 1 << (max(matrix.shape) - 1).bit_length()
    widths = []
    for length in matrix.shape:
        total = side - length
        widths.append((total // 2, total - total // 2))
    padding = Padding(matrix.shape, (side, side), tuple(widths))
    return np.pad(matrix, padding.pad_width, mode=padding.mode), padding


def crop_to_original(matrix: np.ndarray, padding: Padding) -> np.ndarray:
    r0, c0 = padding.pad_width[0][0], padding.pad_width[1][0]
    rows, cols = padding.original_shape
    return np.asarray(matrix)[r0:r0 + rows, c0:c0 + cols]


def make_system(side: int):
    """Construct the one allowed system; ndarray is required by backend 0.0.3."""
    system = pyshearlab.SLgetShearletSystem2D(
        USE_GPU, side, side, N_SCALES, np.asarray(SHEAR_LEVELS), FULL
    )
    if tuple(np.asarray(system["shearLevels"]).tolist()) != SHEAR_LEVELS:
        raise RuntimeError("Backend changed the requested shearLevels")
    if "full" in system and int(system["full"]) != FULL:
        raise RuntimeError(f"Backend changed full: expected {FULL}, found {system['full']}")
    validate_directional_groups(system)
    return system


def directional_groups(system) -> dict[int, np.ndarray]:
    idxs = np.asarray(system["shearletIdxs"], dtype=int)
    if idxs.ndim != 2 or idxs.shape[1] != 3:
        raise ValueError("Invalid shearletIdxs; expected [cone, scale, shearing]")
    return {scale: np.flatnonzero(idxs[:, 1] == scale) for scale in range(1, 4)}


def validate_directional_groups(system) -> dict[int, np.ndarray]:
    idxs = np.asarray(system["shearletIdxs"], dtype=int)
    if "shearLevels" in system and tuple(np.asarray(system["shearLevels"]).tolist()) != SHEAR_LEVELS:
        raise RuntimeError(f"Expected shearLevels {list(SHEAR_LEVELS)}")
    if "full" in system and int(system["full"]) != FULL:
        raise RuntimeError(f"Expected full={FULL}, found {system['full']}")
    if "nShearlets" in system and int(system["nShearlets"]) != EXPECTED_TOTAL_FILTERS:
        raise RuntimeError(
            f"Expected nShearlets={EXPECTED_TOTAL_FILTERS}, found {system['nShearlets']}"
        )
    lowpass = np.flatnonzero(np.all(idxs == LOWPASS_INDEX, axis=1))
    if len(idxs) != EXPECTED_TOTAL_FILTERS:
        raise RuntimeError(f"Expected {EXPECTED_TOTAL_FILTERS} total filters, found {len(idxs)}")
    if len(lowpass) != EXPECTED_LOWPASS_FILTERS:
        raise RuntimeError(f"Expected one low-pass {LOWPASS_INDEX}, found {len(lowpass)}")
    groups = directional_groups(system)
    counts = {scale: len(indices) for scale, indices in groups.items()}
    if counts != EXPECTED_FILTERS_PER_SCALE:
        raise RuntimeError(
            f"Expected directional filters per scale {EXPECTED_FILTERS_PER_SCALE}, found {counts}"
        )
    selected = np.concatenate(list(groups.values()))
    if lowpass[0] in selected or len(selected) != len(idxs) - 1:
        raise RuntimeError("Directional grouping must include every filter except low-pass")
    return groups


def normalized_coefficients(coefficients: np.ndarray, system) -> np.ndarray:
    rms = np.asarray(system["RMS"], dtype=float)
    if coefficients.ndim != 3 or coefficients.shape[2] != len(rms):
        raise ValueError("Coefficient/RMS channel mismatch")
    if not np.isfinite(rms).all() or np.any(rms <= 0):
        raise ValueError("Invalid Shearlet-system RMS values")
    result = coefficients / rms[None, None, :]
    if not np.isfinite(result).all():
        raise ValueError("RMS-normalized coefficients contain NaN/Inf")
    return result


def scale_energy_maps(normalized: np.ndarray, system) -> dict[int, np.ndarray]:
    groups = validate_directional_groups(system)
    maps = {
        scale: np.sqrt(np.sum(np.abs(normalized[:, :, indices]) ** 2, axis=2))
        for scale, indices in groups.items()
    }
    if set(maps) != {1, 2, 3} or not all(np.isfinite(m).all() for m in maps.values()):
        raise RuntimeError("Expected exactly three finite scale-energy maps")
    return maps


def interior_mask_mm(mask: np.ndarray, axis1_mm: np.ndarray, axis2_mm: np.ndarray,
                     distance_mm: float = ROI_DISTANCE_MM) -> tuple[np.ndarray, tuple[float, float]]:
    """Return pixels whose EDT distance to the exterior is strictly above distance_mm."""
    mask = np.asarray(mask)
    axis1_mm, axis2_mm = np.asarray(axis1_mm), np.asarray(axis2_mm)
    if mask.dtype.kind != "b" or mask.shape != (len(axis2_mm), len(axis1_mm)):
        raise ValueError("ROI mask/coordinate shape mismatch")
    spacings = []
    for axis in (axis1_mm, axis2_mm):
        differences = np.diff(axis.astype(float))
        if len(differences) == 0 or np.any(differences <= 0) or not np.allclose(differences, differences[0]):
            raise ValueError("Physical axes must be uniformly and strictly increasing")
        spacings.append(float(differences[0]))
    distance = distance_transform_edt(mask, sampling=(spacings[1], spacings[0]))
    interior = mask & (distance > distance_mm)
    if not interior.any():
        raise ValueError(f"interior_{distance_mm:g}mm ROI is empty")
    return interior, (spacings[0], spacings[1])


def _roi_statistics(values: np.ndarray, mask: np.ndarray, suffix: str) -> dict[str, float]:
    roi = np.asarray(values)[mask]
    if roi.size == 0 or not np.isfinite(roi).all():
        raise ValueError(f"Invalid or empty {suffix} ROI")
    return {
        f"mean_{suffix}": float(np.mean(roi)),
        f"rms_{suffix}": float(np.sqrt(np.mean(roi ** 2))),
        f"p95_{suffix}": float(np.percentile(roi, 95)),
        f"p99_{suffix}": float(np.percentile(roi, 99)),
        f"max_{suffix}": float(np.max(roi)),
        f"total_energy_{suffix}": float(np.sum(roi ** 2)),
    }


def _dominant_direction(normalized: np.ndarray, system, scale: int,
                        roi: np.ndarray) -> dict[str, float | int]:
    idxs = np.asarray(system["shearletIdxs"], dtype=int)
    indices = directional_groups(system)[scale]
    energies = np.asarray([np.sum(np.abs(normalized[:, :, i][roi]) ** 2) for i in indices])
    total = float(energies.sum())
    if total <= 0 or not np.isfinite(total):
        raise ValueError("Directional energy must be positive and finite")
    probabilities = energies / total
    dominant_local = int(np.argmax(energies))
    dominant_idx = indices[dominant_local]
    entropy = 0.0 if len(indices) == 1 else float(
        -np.sum(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0]))
        / np.log(len(indices))
    )
    return {
        "dominant_cone": int(idxs[dominant_idx, 0]),
        "dominant_shearing_index": int(idxs[dominant_idx, 2]),
        "dominant_directional_energy_fraction": float(probabilities[dominant_local]),
        "normalized_directional_entropy": entropy,
    }


def compatibility_smoke_test(size: int = 128) -> dict:
    backend_version()
    yy, xx = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    synthetic = np.exp(-12 * (xx ** 2 + yy ** 2)) + 0.1 * np.sin(9 * xx + 5 * yy)
    system = make_system(size)
    coefficients = pyshearlab.SLsheardec2D(synthetic, system)
    reconstruction = pyshearlab.SLshearrec2D(coefficients, system)
    error = float(np.linalg.norm(reconstruction - synthetic) / np.linalg.norm(synthetic))
    if not np.isfinite(coefficients).all() or not np.isfinite(reconstruction).all():
        raise RuntimeError("Backend smoke test produced NaN/Inf")
    if error > RECONSTRUCTION_TOLERANCE:
        raise RuntimeError(f"Backend reconstruction error {error:.3e} exceeds tolerance")
    return {"shape": [size, size], "n_filters": int(system["nShearlets"]),
            "relative_reconstruction_error": error}


def _git_commit(repository_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _clear_gate2(output_root: Path) -> None:
    root = output_root / "05_shearlet"
    if root.exists():
        for suffix in ("*.npz", "*.png", "*.pdf", "*.csv", "*.json"):
            for path in root.rglob(suffix):
                path.unlink()


def run_shearlet_pipeline(output_root: str | Path, *, git_commit_sha: str | None = None,
                          requested_repo_ref: str = REPO_REF):
    """Analyze the 40 Gate 1 numeric matrices; never reads visualization files."""
    output_root = Path(output_root)
    numeric_root = output_root / "02_numeric"
    paths = sorted(numeric_root.glob("Mode[12]/[XY][ZZ]/*.npz"))
    if len(paths) != EXPECTED_CUTS:
        raise ValueError(f"Expected exactly 40 Gate 1 NPZ matrices, found {len(paths)}")
    gate1_metadata_path = output_root / "04_metrics" / "run_metadata.json"
    gate1_metadata = json.loads(gate1_metadata_path.read_text())
    smoke = compatibility_smoke_test()
    _clear_gate2(output_root)

    systems = {}
    records = []
    level_rows, direction_rows = [], []
    representative_errors = []
    level_roi_values = {1: [], 2: [], 3: []}
    filter_counts = None
    shearlet_idxs = None

    for path in paths:
        with np.load(path, allow_pickle=False) as source:
            required = {"unmasked_intensity_V_per_m", "inside_solid", "axis1_mm", "axis2_mm",
                        "plane", "fixed_mm", "mode", "polarization"}
            if not required.issubset(source.files):
                raise ValueError(f"{path}: missing Gate 1 keys {sorted(required - set(source.files))}")
            field = np.array(source["unmasked_intensity_V_per_m"], dtype=float, copy=True)
            mask = np.array(source["inside_solid"], dtype=bool, copy=True)
            axis1 = np.array(source["axis1_mm"], dtype=float, copy=True)
            axis2 = np.array(source["axis2_mm"], dtype=float, copy=True)
            plane, mode = str(source["plane"]), int(source["mode"])
            fixed_mm, polarization = float(source["fixed_mm"]), str(source["polarization"])
        cut_id = path.stem
        padded, padding = pad_to_power_of_two_square(field)
        side = padding.padded_shape[0]
        if side not in systems:
            systems[side] = make_system(side)
        system = systems[side]
        groups = directional_groups(system)
        counts = {str(scale): int(len(indices)) for scale, indices in groups.items()}
        current_idxs = np.asarray(system["shearletIdxs"], dtype=int).tolist()
        filter_counts = counts if filter_counts is None else filter_counts
        shearlet_idxs = current_idxs if shearlet_idxs is None else shearlet_idxs
        if counts != filter_counts or current_idxs != shearlet_idxs:
            raise RuntimeError("Filter counts changed between cuts")

        raw_coefficients = pyshearlab.SLsheardec2D(padded, system)
        if not np.isfinite(raw_coefficients).all():
            raise RuntimeError(f"{path}: raw coefficients contain NaN/Inf")
        normalized = normalized_coefficients(raw_coefficients, system)
        padded_maps = scale_energy_maps(normalized, system)
        maps = {scale: crop_to_original(array, padding) for scale, array in padded_maps.items()}
        interior, spacing = interior_mask_mm(mask, axis1, axis2)

        reconstruction = pyshearlab.SLshearrec2D(raw_coefficients, system)
        relative_error = float(np.linalg.norm(reconstruction - padded) / np.linalg.norm(padded))
        if not np.isfinite(reconstruction).all() or relative_error > RECONSTRUCTION_TOLERANCE:
            raise RuntimeError(f"{path}: reconstruction error {relative_error:.3e} is unacceptable")
        if cut_id in {"T05", "A05"}:
            representative_errors.append({"mode": mode, "cut_id": cut_id,
                                          "relative_reconstruction_error": relative_error})

        full_totals = {scale: float(np.sum(maps[scale][mask] ** 2)) for scale in range(1, 4)}
        interior_totals = {scale: float(np.sum(maps[scale][interior] ** 2)) for scale in range(1, 4)}
        full_denominator, interior_denominator = sum(full_totals.values()), sum(interior_totals.values())
        if full_denominator <= 0 or interior_denominator <= 0:
            raise ValueError(f"{path}: zero aggregate scale energy")
        base = {"mode": mode, "polarization": polarization, "plane": plane,
                "cut_id": cut_id, "fixed_mm": fixed_mm}
        for scale in range(1, 4):
            row = {**base, "scale": scale,
                   **_roi_statistics(maps[scale], mask, "full_phantom"),
                   **_roi_statistics(maps[scale], interior, "interior_1p5mm"),
                   "scale_energy_fraction_full_phantom": full_totals[scale] / full_denominator,
                   "scale_energy_fraction_interior": interior_totals[scale] / interior_denominator}
            level_rows.append(row)
            padded_full = np.pad(mask, padding.pad_width, mode="constant")
            padded_interior = np.pad(interior, padding.pad_width, mode="constant")
            full_direction = _dominant_direction(normalized, system, scale, padded_full)
            interior_direction = _dominant_direction(normalized, system, scale, padded_interior)
            direction_rows.append({
                **base, "scale": scale,
                **{f"{key}_full_phantom": value for key, value in full_direction.items()},
                **{f"{key}_interior_1p5mm": value for key, value in interior_direction.items()},
                "dominant_direction_changed_interior": (
                    full_direction["dominant_cone"], full_direction["dominant_shearing_index"]
                ) != (
                    interior_direction["dominant_cone"],
                    interior_direction["dominant_shearing_index"],
                ),
            })
            level_roi_values[scale].append(maps[scale][mask])

        artifact = output_root / "05_shearlet" / "numeric" / f"Mode{mode}" / plane / f"{cut_id}.npz"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            artifact, scale_energy_1=maps[1], scale_energy_2=maps[2], scale_energy_3=maps[3],
            inside_solid=mask, interior_1p5mm=interior, axis1_mm=axis1, axis2_mm=axis2,
            original_shape=np.asarray(padding.original_shape), padded_shape=np.asarray(padding.padded_shape),
            pad_width=np.asarray(padding.pad_width), padding_mode=padding.mode,
            input_npz_key="unmasked_intensity_V_per_m", plane=plane, fixed_mm=fixed_mm,
            mode=mode, polarization=polarization,
        )
        records.append({"path": artifact, "mode": mode, "plane": plane, "cut_id": cut_id,
                        "fixed_mm": fixed_mm, "polarization": polarization, "axis1": axis1,
                        "axis2": axis2, "mask": mask, "maps": maps, "padding": padding,
                        "spacing": spacing, "relative_error": relative_error})

    if len(records) != 40 or len(level_rows) != 120 or len(direction_rows) != 120:
        raise RuntimeError("Gate 2 scientific output count mismatch")
    vmax = {scale: float(np.percentile(np.concatenate(level_roi_values[scale]),
                                      VISUALIZATION_PERCENTILE)) for scale in range(1, 4)}
    if any(not np.isfinite(value) or value <= 0 for value in vmax.values()):
        raise ValueError("Invalid global per-level visualization limits")

    for record in records:
        for scale in range(1, 4):
            destination = (output_root / "05_shearlet" / "level_maps" /
                           f"Mode{record['mode']}" / record["plane"] / record["cut_id"] /
                           f"level_{scale}.png")
            save_level_map(record["axis1"], record["axis2"], record["maps"][scale],
                           record["mask"], destination,
                           f"{record['cut_id']} · Mode {record['mode']} · Level {scale}", vmax[scale])

    metrics_root = output_root / "05_shearlet" / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    for name, rows in (("shearlet_level_metrics.csv", level_rows),
                       ("shearlet_direction_metrics.csv", direction_rows)):
        with (metrics_root / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)

    figures = output_root / "05_shearlet" / "article_figures"
    for cut_id in ("T05", "A05"):
        selected = [r for r in records if r["cut_id"] == cut_id]
        for suffix in ("png", "pdf"):
            save_representative_levels(selected, figures / f"representative_{cut_id}_3levels.{suffix}", vmax)
    for suffix in ("png", "pdf"):
        save_fraction_comparison(level_rows, figures / f"scale_energy_comparison.{suffix}")

    metadata = {
        "backend_name": BACKEND_DISTRIBUTION, "backend_import": BACKEND_IMPORT,
        "backend_version": backend_version(), "useGPU": USE_GPU, "nScales": N_SCALES,
        "shearLevels": list(SHEAR_LEVELS), "full": FULL,
        "expected_total_filters": EXPECTED_TOTAL_FILTERS,
        "actual_total_filters": len(shearlet_idxs),
        "expected_filters_per_scale": {str(k): v for k, v in EXPECTED_FILTERS_PER_SCALE.items()},
        "actual_filters_per_scale": filter_counts,
        "shearletIdxs": shearlet_idxs,
        "scale_interpretation": {str(k): v for k, v in SCALE_INTERPRETATION.items()},
        "coefficient_normalization": "C_normalized[:,:,k] = C_raw[:,:,k] / shearletSystem['RMS'][k]",
        "scale_energy_formula": "sqrt(sum_k(abs(C_normalized_jk)**2)); [0,0,0] low-pass excluded",
        "coefficient_energy_interpretation": (
            "Shearlet coefficient energy is a signal-processing quantity based on squared "
            "RMS-normalized Shearlet coefficients. It is not electromagnetic energy and is "
            "not Poynting intensity."
        ),
        "directional_entropy_formula": "H = -sum_k(p_k ln p_k) / ln(K), p_k = E_k/sum_q(E_q)",
        "padding_method": "symmetric reflect padding to smallest power-of-two square; no resize/interpolation",
        "padded_sizes": sorted({r["padding"].padded_shape[0] for r in records}),
        "field_quantity": "Electric-field magnitude |E|", "units": "V/m",
        "input_npz_key": "unmasked_intensity_V_per_m",
        "roi_definitions": {
            "full_phantom": "Gate 1 inside_solid mask",
            "interior_1p5mm": "inside_solid pixels with Euclidean EDT distance > 1.5 mm; sampling=(axis2 spacing, axis1 spacing)",
        },
        "visualization_percentile": VISUALIZATION_PERCENTILE,
        "visualization_vmax_per_level": {str(k): v for k, v in vmax.items()},
        "visualization_colormap": "cividis", "visualization_outside_color": "#f2f2f2",
        "requested_repo_ref": requested_repo_ref,
        "git_commit_sha": git_commit_sha or _git_commit(Path(__file__).resolve().parents[1]),
        "software_versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
            "pyShearLab-MIND": backend_version(),
        },
        "dataset_filename": gate1_metadata.get("dataset_filename"),
        "dataset_sha256": gate1_metadata.get("dataset_sha256"),
        "number_of_cuts": len(records), "number_of_numeric_scale_maps": len(records) * 3,
        "number_of_level_maps": len(list((output_root / "05_shearlet" / "level_maps").rglob("*.png"))),
        "system_sizes": sorted({r["padding"].padded_shape[0] for r in records}),
        "filters_per_scale": filter_counts, "lowpass_filters": 1,
        "article_figure_artifact_paths": [
            str(Path("05_shearlet/article_figures") / f"representative_{cut}_3levels.{suffix}")
            for cut in ("T05", "A05") for suffix in ("png", "pdf")
        ] + [str(Path("05_shearlet/article_figures") / f"scale_energy_comparison.{suffix}")
             for suffix in ("png", "pdf")],
        "smoke_test": smoke, "representative_reconstruction_results": representative_errors,
        "maximum_all_cut_reconstruction_error": max(r["relative_error"] for r in records),
        "reconstruction_tolerance": RECONSTRUCTION_TOLERANCE,
    }
    (metrics_root / "shearlet_run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if metadata["number_of_level_maps"] != EXPECTED_LEVEL_MAPS:
        raise RuntimeError(f"Expected 120 rendered level maps, got {metadata['number_of_level_maps']}")
    return level_rows
