from pathlib import Path

import numpy as np
import pytest

from src.fld_reader import pair_real_imag, read_fld


def _write_fld(path: Path, rows: str, *, descriptor: str = "Real(<Ex,Ey,Ez>)", fixed_max: str = "1.0E+00") -> None:
    path.write_text(
        (
            "  Grid Output Min: [ -1.0E+00mm, 0.0mm, -2.5e+00mm ]   "
            f"Max: [ {fixed_max}mm, 0.0mm, 2.5E+00mm ]  "
            "Grid Size: [ 1e0mm, 0mm, 2.5e0mm ] Unit: \"mm\"\n"
            + descriptor + "\n" + rows
        ),
        encoding="utf-8",
    )


def test_parser_accepts_whitespace_commas_and_scientific_notation(tmp_path):
    path = tmp_path / "sample.fld"
    _write_fld(path, "  -1e0  0  -2.5E+0   1.2e-3  -2E+1  3.0 \n")
    field = read_fld(path)
    assert field.coord_unit == "mm"
    assert field.grid_min == (-1.0, 0.0, -2.5)
    assert field.vectors.shape == (1, 3)
    assert np.allclose(field.vectors[0], [1.2e-3, -20, 3])


def test_parser_and_pairing_fail_clearly_on_bad_data(tmp_path):
    real = tmp_path / "real.fld"
    imag = tmp_path / "imag.fld"
    _write_fld(real, "0 0 0 1 2 3\n")
    _write_fld(imag, "1 0 0 1 2 3\n", descriptor="Imag(<Ex,Ey,Ez>)")
    with pytest.raises(ValueError, match="Coordinate mismatch"):
        pair_real_imag(real, imag)

    _write_fld(imag, "0 0 0 nan 2 3\n", descriptor="Imag(<Ex,Ey,Ez>)")
    with pytest.raises(ValueError, match="NaN/Inf"):
        pair_real_imag(real, imag)


def test_pairing_validates_real_and_imag_descriptors(tmp_path):
    real = tmp_path / "real.fld"
    imag = tmp_path / "imag.fld"
    rows = "0 0 0 1 2 3\n"
    _write_fld(real, rows, descriptor="Imag(<Ex,Ey,Ez>)")
    _write_fld(imag, rows, descriptor="Imag(<Ex,Ey,Ez>)")
    with pytest.raises(ValueError, match="real file descriptor"):
        pair_real_imag(real, imag)

    _write_fld(real, rows, descriptor="Real(<Ex,Ey,Ez>)")
    _write_fld(imag, rows, descriptor="Real(<Ex,Ey,Ez>)")
    with pytest.raises(ValueError, match="imag file descriptor"):
        pair_real_imag(real, imag)
