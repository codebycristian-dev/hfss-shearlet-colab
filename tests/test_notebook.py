import json
from pathlib import Path


def test_colab_notebook_public_clone_and_gate1_assertions_are_reproducible():
    notebook = json.loads(Path("notebook/HFSS_Shearlet_2p45GHz.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert 'REPO_URL = "https://github.com/codebycristian-dev/hfss-shearlet-colab.git"' in source
    assert 'if IN_COLAB and not (REPO_ROOT / "src").is_dir():' in source
    assert '["git", "clone", "--depth", "1", REPO_URL, str(REPO_ROOT)]' in source
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
