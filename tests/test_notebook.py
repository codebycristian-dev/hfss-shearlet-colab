import json
from pathlib import Path


def test_colab_notebook_private_clone_and_gate1_assertions_are_reproducible():
    notebook = json.loads(Path("notebook/HFSS_Shearlet_2p45GHz.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert 'userdata.get("GITHUB_TOKEN")' in source
    assert 'REPOSITORY_IS_PRIVATE = True' in source
    assert '"pip", "install", "-r"' in source
    assert "hashlib.sha256()" in source
    assert 'DATA_ZIP.open("rb")' in source
    assert "dataset_sha256=DATASET_SHA256" in source
    assert '"git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"' in source
    assert "GITHUB_TOKEN@github.com" not in source
    assert "Expected 40 physical PNGs" in source
    assert "Expected 40 presentation PNGs" in source
    assert "Expected 40 numeric matrices" in source
    assert "Expected 40 metric rows" in source
    assert "parallel_y" in source and "perpendicular_y" in source
