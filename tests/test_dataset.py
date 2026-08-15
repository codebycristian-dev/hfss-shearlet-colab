from pathlib import Path

import pytest

from src.dataset import TARGET_DIRECTORIES, discover_phantom_pairs, validate_expected_40


XZ = (-90, -70, -50, -30, -10, 10, 30, 50, 70, 90)
YZ = (-45, -35, -25, -15, -5, 5, 15, 25, 35, 45)


def _touch_expected(root: Path) -> None:
    for mode in (1, 2):
        xz_root = root / TARGET_DIRECTORIES[(mode, "XZ")]
        yz_root = root / TARGET_DIRECTORIES[(mode, "YZ")]
        xz_root.mkdir(parents=True, exist_ok=True)
        yz_root.mkdir(parents=True, exist_ok=True)
        for index, fixed in enumerate(XZ, 1):
            for part in ("real", "imag"):
                (xz_root / f"E_{part}_T{index:02d}_XZ_y{fixed:+04d}_M{mode}_2p45GHz.fld").touch()
        for index, fixed in enumerate(YZ, 1):
            for part in ("real", "imag"):
                (yz_root / f"E_{part}_A{index:02d}_YZ_x{fixed:+04d}_M{mode}_2p45GHz.fld").touch()


def test_expected_dataset_discovery_is_exact(tmp_path):
    _touch_expected(tmp_path)
    pairs = discover_phantom_pairs(tmp_path)
    validate_expected_40(pairs)
    assert len(pairs) == 40
    # Legitimate production files outside the four target directories are ignored.
    for relative in (
        "01_Phantom_Mode1/00_validation/check.fld",
        "01_Phantom_Mode1/03_aperture_XY/aperture.fld",
        "02_Phantom_Mode2/03_aperture_XY/aperture.fld",
        "baselines/free_space/baseline.fld",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert len(discover_phantom_pairs(tmp_path)) == 40

    unexpected = tmp_path / TARGET_DIRECTORIES[(1, "XZ")] / "unexpected.fld"
    unexpected.touch()
    with pytest.raises(ValueError, match="Unrecognized"):
        discover_phantom_pairs(tmp_path)


def test_discovery_rejects_missing_duplicate_and_mislabeled_cut(tmp_path):
    _touch_expected(tmp_path)
    target = tmp_path / TARGET_DIRECTORIES[(1, "XZ")]
    missing = target / "E_imag_T01_XZ_y-090_M1_2p45GHz.fld"
    missing.unlink()
    with pytest.raises(ValueError, match="Missing real/imag"):
        discover_phantom_pairs(tmp_path)

    missing.touch()
    duplicate_dir = target / "duplicate"
    duplicate_dir.mkdir()
    (duplicate_dir / missing.name).touch()
    with pytest.raises(ValueError, match="Duplicate imag"):
        discover_phantom_pairs(tmp_path)

    (duplicate_dir / missing.name).unlink()
    wrong = target / "E_real_T01_XZ_y-090_M1_2p45GHz.fld"
    wrong.rename(target / "E_real_T01_XZ_y-089_M1_2p45GHz.fld")
    missing.rename(target / "E_imag_T01_XZ_y-089_M1_2p45GHz.fld")
    with pytest.raises(ValueError, match="invalid XZ cuts"):
        validate_expected_40(discover_phantom_pairs(tmp_path))
