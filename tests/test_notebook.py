import json
from pathlib import Path


def test_colab_notebook_public_clone_and_gate1_assertions_are_reproducible():
    notebook = json.loads(Path("notebook/HFSS_Shearlet_2p45GHz.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert 'REPO_URL = "https://github.com/codebycristian-dev/hfss-shearlet-colab.git"' in source
    assert 'if IN_COLAB and not (REPO_ROOT / "src").is_dir():' in source
    assert 'REPO_REF = "gate2-shearlet"' in source
    assert '["git", "clone", "--depth", "1", "--branch", REPO_REF, "--single-branch", REPO_URL, str(REPO_ROOT)]' in source
    assert '["git", "clone", "--depth", "1", REPO_URL' not in source
    assert 'change REPO_REF back to "main" immediately before merging' in source
    for private_auth_term in (
        "GITHUB_TOKEN",
        "google.colab import userdata",
        "GIT_ASKPASS",
        "askpass",
        "GIT_TERMINAL_PROMPT",
        "REPOSITORY_IS_PRIVATE",
    ):
        assert private_auth_term not in source
    assert '"pip", "install", "-r"' in source
    assert "hashlib.sha256()" in source
    assert 'DATA_ZIP.open("rb")' in source
    assert "dataset_sha256=DATASET_SHA256" in source
    assert '"git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"' in source
    assert "Expected 40 physical PNGs" in source
    assert "Expected 40 presentation PNGs" in source
    assert "Expected 40 article PNGs" in source
    assert "Expected 6 article figures" in source
    assert 'glob("*.pdf")' in source
    assert "Expected 40 numeric matrices" in source
    assert "Expected 40 metric rows" in source
    assert "parallel_y" in source and "perpendicular_y" in source
    assert "shared `cividis` scale" in source
    assert "numeric NPZ matrices" in source


def test_notebook_runs_gate2_after_gate1_and_packages_combined_outputs():
    notebook = json.loads(Path("notebook/HFSS_Shearlet_2p45GHz.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert source.index("GATE 1: PASS") < source.index("run_shearlet_pipeline")
    assert source.index("GATE 2: PASS") < source.index("HFSS_Gate1_Gate2_results_2p45GHz")
    assert "unmasked_intensity_V_per_m" in source
    assert "[1, 1, 2]" in source
    assert "Euclidean distance transform" in source
    assert "Expected 120 Shearlet level maps" in source
    assert "Coefficient energy means a discrete sum" in source
    assert "coarser spatial-scale band" in source and "finer spatial-scale band" in source
    assert "dominant_cone_full_phantom" in source
    assert "dominant_cone_interior_1p5mm" in source


def test_user_facing_shearlet_terminology_is_physically_unambiguous():
    readme = Path("README.md").read_text()
    visualization = Path("src/shearlet_visualization.py").read_text()
    assert "It is not electromagnetic energy and is not Poynting intensity" in readme
    assert "RMS-normalized Shearlet response" in visualization
    assert "RMS-normalized scale energy" not in visualization
    assert 'for ax, plane in zip(axes, ("XZ", "YZ"))' in visualization
    assert "scale_energy_fraction_interior" in visualization
