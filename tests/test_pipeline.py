import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.dataset import FieldPair, TARGET_DIRECTORIES
from src.pipeline import _validate_cut_coordinates, run_intensity_pipeline


XZ = (-90, -70, -50, -30, -10, 10, 30, 50, 70, 90)
YZ = (-45, -35, -25, -15, -5, 5, 15, 25, 35, 45)


def _write(path: Path, coords, vector_value: float, descriptor: str, *, spike: bool = False) -> None:
    array = np.column_stack((coords, np.full((len(coords), 3), vector_value)))
    if spike:
        array[len(array) // 2, 3] = 100.0
    lines = [" ".join(f"{value:.9E}" for value in row) for row in array]
    path.write_text(
        (
            "Grid Output Min: [0mm 0mm 0mm] Max: [1mm 1mm 1mm] "
            "Grid Size: [1mm 1mm 1mm] Unit: \"mm\"\n"
            + descriptor + "\n" + "\n".join(lines) + "\n"
        ),
        encoding="utf-8",
    )


def _dataset(root: Path) -> None:
    for mode in (1, 2):
        xz_root = root / TARGET_DIRECTORIES[(mode, "XZ")]
        yz_root = root / TARGET_DIRECTORIES[(mode, "YZ")]
        xz_root.mkdir(parents=True, exist_ok=True)
        yz_root.mkdir(parents=True, exist_ok=True)
        for index, fixed in enumerate(XZ, 1):
            coords = np.array([(x, fixed, z) for z in (-50., 0., 50.) for x in (-50., 0., 50.)])
            for part, value in (("real", mode), ("imag", 0)):
                descriptor = f"{part.title()}(<Ex,Ey,Ez>)"
                _write(
                    xz_root / f"E_{part}_T{index:02d}_XZ_y{fixed:+04d}_M{mode}_2p45GHz.fld",
                    coords, value, descriptor,
                    spike=(mode == 2 and index == 10 and part == "real"),
                )
        for index, fixed in enumerate(YZ, 1):
            coords = np.array([(fixed, y, z) for z in (-20., 0., 20.) for y in (-100., 0., 100.)])
            for part, value in (("real", mode), ("imag", 0)):
                descriptor = f"{part.title()}(<Ex,Ey,Ez>)"
                _write(yz_root / f"E_{part}_A{index:02d}_YZ_x{fixed:+04d}_M{mode}_2p45GHz.fld", coords, value, descriptor)


def test_pipeline_writes_exact_outputs_numeric_matrices_and_metrics(tmp_path, monkeypatch):
    data = tmp_path / "data"
    output = tmp_path / "outputs"
    data.mkdir()
    _dataset(data)

    plot_calls = []
    article_calls = []
    mosaic_calls = []
    def fake_plot(axis1, axis2, intensity, out_path, title, *args, **kwargs):
        assert intensity.ndim == 2
        assert kwargs["vmin"] == 0.0
        assert title.count("\n") == 1
        assert "Electric-field magnitude |E| · 2.45 GHz" in title
        assert "parallel_y" not in title and "perpendicular_y" not in title
        plot_calls.append((Path(out_path), kwargs["vmax"], title))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"PNG")

    monkeypatch.setattr("src.pipeline.save_intensity_image", fake_plot)
    def fake_article(axis1, axis2, intensity, mask, out_path, title, *args, **kwargs):
        assert intensity.ndim == mask.ndim == 2
        assert intensity.shape == mask.shape
        assert kwargs["vmin"] == 0.0
        article_calls.append((Path(out_path), kwargs["vmax"], title))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"PNG")
    def fake_mosaic(panels, out_path, **kwargs):
        assert kwargs["vmin"] == 0.0
        mosaic_calls.append((Path(out_path), panels, kwargs))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"PNG")
    monkeypatch.setattr("src.pipeline.save_article_image", fake_article)
    monkeypatch.setattr("src.pipeline.save_article_mosaic", fake_mosaic)
    stale = output / "01_intensity" / "old.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")

    dataset_hash = "ab" * 32
    rows = run_intensity_pipeline(
        data, output,
        dataset_filename="Entregas_papper(1).zip",
        dataset_sha256=dataset_hash,
        git_commit_sha="1234567890abcdef",
    )
    physical_pngs = list((output / "01_intensity" / "physical_shared").rglob("*.png"))
    presentation_pngs = list((output / "01_intensity" / "presentation_shared").rglob("*.png"))
    article_pngs = list((output / "01_intensity" / "article_shared").rglob("*.png"))
    matrices = list((output / "02_numeric").rglob("*.npz"))
    with (output / "04_metrics" / "field_metrics.csv").open(newline="") as stream:
        metrics = list(csv.DictReader(stream))

    assert len(rows) == len(physical_pngs) == len(presentation_pngs) == len(article_pngs) == len(matrices) == len(metrics) == 40
    assert len(plot_calls) == 80
    assert len(article_calls) == 40
    assert len(mosaic_calls) == 6
    assert not stale.exists()
    assert {(p.parent.parent.name, p.parent.name) for p in physical_pngs} == {
        ("Mode1", "XZ"), ("Mode1", "YZ"), ("Mode2", "XZ"), ("Mode2", "YZ")
    }
    sample = np.load(matrices[0])
    assert sample["intensity_V_per_m"].ndim == 2
    assert sample["intensity_V_per_m"].dtype.kind == "f"
    assert set(sample.files) == {
        "intensity_V_per_m", "unmasked_intensity_V_per_m", "inside_solid",
        "axis1_mm", "axis2_mm", "plane", "fixed_mm", "mode", "polarization",
    }
    assert np.isclose(float(metrics[0]["E_max_V_per_m"]), np.sqrt(3))
    assert set(row["polarization"] for row in metrics) == {"parallel_y", "perpendicular_y"}
    assert str(sample["polarization"]) in {"parallel_y", "perpendicular_y"}
    metadata = json.loads((output / "04_metrics" / "run_metadata.json").read_text())
    assert metadata["dataset_filename"] == "Entregas_papper(1).zip"
    assert metadata["dataset_sha256"] == dataset_hash
    assert metadata["git_commit_sha"] == "1234567890abcdef"
    assert metadata["frequency_GHz"] == 2.45
    assert metadata["quantity"] == "Electric-field magnitude |E|"
    assert metadata["unit"] == "V/m"
    assert metadata["phantom_radius_mm"] == 50.0
    assert metadata["phantom_length_mm"] == 200.0
    assert metadata["XZ_cut_positions_mm"] == list(XZ)
    assert metadata["YZ_cut_positions_mm"] == list(YZ)
    assert metadata["presentation_percentile"] == 99.5
    assert metadata["presentation_shared_vmax_V_per_m"] < metadata["physical_shared_vmax_V_per_m"]
    assert metadata["article_colormap"] == "cividis"
    assert metadata["article_scale_type"] == "shared global percentile"
    assert metadata["article_vmax_V_per_m"] == metadata["presentation_shared_vmax_V_per_m"]
    assert metadata["article_outside_mask_color"] == "#f2f2f2"
    assert metadata["article_figures"] == [
        "03_article_figures/xz_mosaic_all.png",
        "03_article_figures/xz_mosaic_all.pdf",
        "03_article_figures/yz_mosaic_all.png",
        "03_article_figures/yz_mosaic_all.pdf",
        "03_article_figures/representative_2x2.png",
        "03_article_figures/representative_2x2.pdf",
    ]
    assert metadata["polarization_by_mode"] == {"Mode1": "parallel_y", "Mode2": "perpendicular_y"}
    physical_limits = {limit for path, limit, _ in plot_calls if "physical_shared" in path.parts}
    presentation_limits = {limit for path, limit, _ in plot_calls if "presentation_shared" in path.parts}
    article_limits = {limit for _, limit, _ in article_calls}
    assert physical_limits == {metadata["physical_shared_vmax_V_per_m"]}
    assert presentation_limits == {metadata["presentation_shared_vmax_V_per_m"]}
    assert article_limits == {metadata["presentation_shared_vmax_V_per_m"]}
    assert {path.name for path, _, _ in mosaic_calls} == {
        "xz_mosaic_all.png", "xz_mosaic_all.pdf",
        "yz_mosaic_all.png", "yz_mosaic_all.pdf",
        "representative_2x2.png", "representative_2x2.pdf",
    }
    assert [(call[2]["nrows"], call[2]["ncols"], len(call[1])) for call in mosaic_calls] == [
        (4, 5, 20), (4, 5, 20), (4, 5, 20), (4, 5, 20), (2, 2, 4), (2, 2, 4),
    ]
    for _, panels, options in mosaic_calls[:4]:
        assert [panel["title"] for panel in panels] == [
            *(f"{panels[0]['title'][0]}{index:02d}" for index in range(1, 11)),
            *(f"{panels[0]['title'][0]}{index:02d}" for index in range(1, 11)),
        ]
        assert options["row_group_labels"] == [
            "Mode 1 · parallel to y", "Mode 1 · parallel to y",
            "Mode 2 · perpendicular to y", "Mode 2 · perpendicular to y",
        ]
    article_files = sorted((output / "03_article_figures").iterdir())
    assert len(article_files) == 6
    assert {path.suffix for path in article_files} == {".png", ".pdf"}
    outlier_matrix = np.load(output / "02_numeric" / "Mode2" / "XZ" / "T10.npz")
    assert outlier_matrix["intensity_V_per_m"].max() == metadata["physical_shared_vmax_V_per_m"]
    assert outlier_matrix["intensity_V_per_m"].max() > metadata["presentation_shared_vmax_V_per_m"]
    titles = [title for _, _, title in plot_calls]
    assert any("T05 · XZ · y = -10 mm · Mode 1 (parallel to y)" in title for title in titles)
    assert any("A05 · YZ · x = -5 mm · Mode 1 (parallel to y)" in title for title in titles)
    assert any("Mode 2 (perpendicular to y)" in title for title in titles)
    assert any(title.endswith("physical shared scale") for title in titles)
    assert any(title.endswith("presentation shared scale") for title in titles)


def test_pipeline_rejects_wrong_fixed_cut_coordinate(tmp_path):
    pair = FieldPair(1, "XZ", 1, -90, tmp_path / "real.fld", tmp_path / "imag.fld")
    coords = np.array([[0., -89., 0.], [1., -89., 1.]])
    with pytest.raises(ValueError, match="expected fixed y=-90"):
        _validate_cut_coordinates(pair, coords, "mm")
