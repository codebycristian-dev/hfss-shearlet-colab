from importlib.metadata import version
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
import pytest

from src import shearlet_analysis as sa


def _system():
    idxs = []
    for scale, count in sa.EXPECTED_FILTERS_PER_SCALE.items():
        idxs.extend([[1 + (i % 2), scale, i - count // 2] for i in range(count)])
    idxs.append([0, 0, 0])
    return {
        "shearletIdxs": np.asarray(idxs),
        "RMS": np.arange(1, 34, dtype=float),
        "shearLevels": np.array([1, 1, 2]),
        "full": 0,
    }


def test_exact_backend_compatibility_and_reconstruction():
    assert version("pyShearLab-MIND") == "0.0.3"
    result = sa.compatibility_smoke_test(128)
    assert result["n_filters"] == 33
    assert result["relative_reconstruction_error"] < sa.RECONSTRUCTION_TOLERANCE


def test_exact_configuration_and_shearlet_idx_grouping_excludes_lowpass():
    assert (sa.USE_GPU, sa.N_SCALES, sa.SHEAR_LEVELS, sa.FULL) == (0, 3, (1, 1, 2), 0)
    groups = sa.validate_directional_groups(_system())
    assert set(groups) == {1, 2, 3}
    assert [len(indices) for indices in groups.values()] == [8, 8, 16]
    assert 32 not in np.concatenate(list(groups.values()))


def test_production_system_rejects_wrong_total_or_scale_counts():
    system = _system()
    system["shearletIdxs"] = system["shearletIdxs"][:-1]
    with pytest.raises(RuntimeError, match="33 total filters"):
        sa.validate_directional_groups(system)
    system = _system(); system["full"] = 1
    with pytest.raises(RuntimeError, match="full=0"):
        sa.validate_directional_groups(system)


def test_padding_crop_roundtrip_has_no_resize_or_interpolation():
    original = np.arange(15, dtype=float).reshape(3, 5)
    padded, info = sa.pad_to_power_of_two_square(original)
    assert padded.shape == (8, 8)
    assert info.pad_width == ((2, 3), (1, 2))
    assert info.mode == "reflect"
    assert np.array_equal(sa.crop_to_original(padded, info), original)


def test_rms_normalization_and_exactly_three_energy_maps():
    raw = np.ones((2, 3, 33)) * np.arange(1, 34)
    normalized = sa.normalized_coefficients(raw, _system())
    assert np.allclose(normalized[0, 0], np.arange(1, 34) / _system()["RMS"])
    maps = sa.scale_energy_maps(normalized, _system())
    assert set(maps) == {1, 2, 3}
    expected_level_1 = np.sqrt(np.sum(normalized[:, :, :8] ** 2, axis=2))
    assert np.allclose(maps[1], expected_level_1)
    # Altering only low-pass must not alter any scale map.
    changed = normalized.copy(); changed[:, :, 32] = 1e12
    for scale in maps:
        assert np.array_equal(sa.scale_energy_maps(changed, _system())[scale], maps[scale])


def test_interior_roi_uses_physical_euclidean_distance():
    mask = np.ones((11, 7), dtype=bool)
    mask[[0, -1], :] = False; mask[:, [0, -1]] = False
    axis1 = np.arange(7) * 1.0
    axis2 = np.arange(11) * 0.5
    interior, spacing = sa.interior_mask_mm(mask, axis1, axis2, 1.5)
    assert spacing == (1.0, 0.5)
    assert interior[5, 3]
    assert not interior[2, 3]  # 1.0 mm vertically from exterior.


def test_directional_entropy_is_normalized_and_finite():
    normalized = np.ones((4, 4, 33))
    roi = np.ones((4, 4), dtype=bool)
    result = sa._dominant_direction(normalized, _system(), 1, roi)
    assert result["dominant_directional_energy_fraction"] == pytest.approx(1 / 8)
    assert result["normalized_directional_entropy"] == pytest.approx(1.0)


def test_directional_results_are_computed_independently_for_both_rois():
    normalized = np.ones((4, 4, 33))
    normalized[0, :, 0] = 20
    full = np.ones((4, 4), dtype=bool)
    interior = np.zeros((4, 4), dtype=bool); interior[1:3, 1:3] = True
    full_result = sa._dominant_direction(normalized, _system(), 1, full)
    interior_result = sa._dominant_direction(normalized, _system(), 1, interior)
    assert full_result["dominant_directional_energy_fraction"] > interior_result[
        "dominant_directional_energy_fraction"
    ]


def test_source_selects_unmasked_numeric_npz_not_png_or_masked_input():
    source = Path("src/shearlet_analysis.py").read_text()
    assert 'source["unmasked_intensity_V_per_m"]' in source
    assert 'source["intensity_V_per_m"]' not in source
    assert "imread" not in source and "PIL" not in source
    assert sa.EXPECTED_CUTS == 40 and sa.EXPECTED_LEVEL_MAPS == 120


def test_visualization_requires_shared_vmax(monkeypatch, tmp_path):
    from src.shearlet_visualization import save_level_map
    with pytest.raises(ValueError, match="global per-level vmax"):
        save_level_map(np.arange(2), np.arange(2), np.ones((2, 2)),
                       np.ones((2, 2), dtype=bool), tmp_path / "x.png", "T01", None)


def test_pipeline_deterministically_writes_40_artifacts_and_120_maps_and_rows(tmp_path, monkeypatch):
    output = tmp_path / "outputs"
    axes = np.arange(7, dtype=float)
    mask = np.ones((7, 7), dtype=bool); mask[[0, -1], :] = False; mask[:, [0, -1]] = False
    originals = {}
    for mode in (1, 2):
        for plane, prefix in (("XZ", "T"), ("YZ", "A")):
            for index in range(1, 11):
                path = output / "02_numeric" / f"Mode{mode}" / plane / f"{prefix}{index:02d}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                unmasked = np.full((7, 7), mode + index / 100, dtype=float)
                unmasked[~mask] += 50  # proves the full plane, not zero-masked input, is decomposed
                np.savez_compressed(
                    path, unmasked_intensity_V_per_m=unmasked,
                    intensity_V_per_m=np.where(mask, unmasked, 0), inside_solid=mask,
                    axis1_mm=axes, axis2_mm=axes, plane=plane, fixed_mm=index,
                    mode=mode, polarization="parallel_y" if mode == 1 else "perpendicular_y",
                )
                originals[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = output / "04_metrics" / "run_metadata.json"
    metadata.parent.mkdir(parents=True); metadata.write_text(json.dumps({"dataset_sha256": "ab" * 32}))

    system = _system(); system.update({"nShearlets": 33, "size": np.array([8, 8]),
                                      "full": 0, "useGPU": 0})
    monkeypatch.setattr(sa, "backend_version", lambda: "0.0.3")
    monkeypatch.setattr(sa, "compatibility_smoke_test", lambda: {
        "shape": [8, 8], "n_filters": 33, "relative_reconstruction_error": 0.0})
    monkeypatch.setattr(sa, "make_system", lambda side: system)
    def decompose(array, _system):
        coefficients = np.repeat(array[:, :, None], 33, axis=2)
        return coefficients
    monkeypatch.setattr(sa.pyshearlab, "SLsheardec2D", decompose)
    monkeypatch.setattr(sa.pyshearlab, "SLshearrec2D", lambda coefficients, _system: coefficients[:, :, 0])
    def render_level(axis1, axis2, values, roi, out_path, title, vmax):
        assert values.shape == roi.shape == (7, 7)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True); Path(out_path).write_bytes(b"PNG")
    monkeypatch.setattr(sa, "save_level_map", render_level)
    monkeypatch.setattr(sa, "save_representative_levels",
                        lambda records, path, vmax: (Path(path).parent.mkdir(parents=True, exist_ok=True), Path(path).write_bytes(b"FIG")))
    monkeypatch.setattr(sa, "save_fraction_comparison",
                        lambda rows, path: (Path(path).parent.mkdir(parents=True, exist_ok=True), Path(path).write_bytes(b"FIG")))

    rows = sa.run_shearlet_pipeline(output, git_commit_sha="deadbeef")
    assert len(rows) == 120
    assert len(list((output / "05_shearlet/numeric").rglob("*.npz"))) == 40
    assert len(list((output / "05_shearlet/level_maps").rglob("*.png"))) == 120
    with (output / "05_shearlet/metrics/shearlet_level_metrics.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 120
    with (output / "05_shearlet/metrics/shearlet_direction_metrics.csv").open(newline="") as stream:
        direction_rows = list(csv.DictReader(stream))
        assert len(direction_rows) == 120
        assert "dominant_cone_full_phantom" in direction_rows[0]
        assert "dominant_cone_interior_1p5mm" in direction_rows[0]
        assert "roi" not in direction_rows[0]
    sample = np.load(output / "05_shearlet/numeric/Mode1/XZ/T01.npz")
    assert {f"scale_energy_{i}" for i in (1, 2, 3)}.issubset(sample.files)
    assert str(sample["input_npz_key"]) == "unmasked_intensity_V_per_m"
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in originals.items())
    run_metadata = json.loads((output / "05_shearlet/metrics/shearlet_run_metadata.json").read_text())
    assert run_metadata["actual_total_filters"] == 33
    assert run_metadata["actual_filters_per_scale"] == {"1": 8, "2": 8, "3": 16}
    assert len(run_metadata["shearletIdxs"]) == 33
    assert run_metadata["requested_repo_ref"] == "main"
    assert run_metadata["dataset_filename"] is None
    assert run_metadata["scale_interpretation"]["3"] == "finer spatial-scale band"
    assert set(run_metadata["software_versions"]) == {
        "python", "numpy", "scipy", "matplotlib", "pyShearLab-MIND"
    }


def test_article_comparison_separates_planes_rois_and_uses_ten_cut_statistics(tmp_path):
    from src.shearlet_visualization import save_fraction_comparison
    rows = []
    for plane in ("XZ", "YZ"):
        for mode in (1, 2):
            for cut in range(10):
                for scale in (1, 2, 3):
                    rows.append({"plane": plane, "mode": mode, "scale": scale,
                                 "scale_energy_fraction_full_phantom": scale / 6,
                                 "scale_energy_fraction_interior": (4 - scale) / 6})
    save_fraction_comparison(rows, tmp_path / "comparison.png")
    assert (tmp_path / "comparison.png").is_file()
