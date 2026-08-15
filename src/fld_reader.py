
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np

_HEADER_RE = re.compile(
    r'Grid\s+Output\s+Min\s*:\s*\[([^\]]+)\]\s*'
    r'Max\s*:\s*\[([^\]]+)\]\s*Grid\s+Size\s*:\s*\[([^\]]+)\]\s*'
    r'Unit\s*:\s*"([^"]+)"', re.IGNORECASE
)
_NUMBER_RE = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
)
_REAL_DESCRIPTOR_RE = re.compile(r"\bReal\s*\(\s*<\s*Ex\s*,\s*Ey\s*,\s*Ez\s*>\s*\)", re.IGNORECASE)
_IMAG_DESCRIPTOR_RE = re.compile(r"\bImag\s*\(\s*<\s*Ex\s*,\s*Ey\s*,\s*Ez\s*>\s*\)", re.IGNORECASE)

def _parse_triplet(text: str) -> tuple[float, float, float]:
    vals = []
    for token in re.split(r"[\s,]+", text.strip()):
        match = _NUMBER_RE.match(token)
        if not match:
            raise ValueError(f"Invalid numeric value {token!r} in {text!r}")
        vals.append(float(match.group().replace("D", "E").replace("d", "e")))
    if len(vals) != 3:
        raise ValueError(f"Expected 3 values, got {text!r}")
    return tuple(vals)

@dataclass(frozen=True)
class FLDData:
    path: Path
    grid_min: tuple[float, float, float]
    grid_max: tuple[float, float, float]
    grid_step: tuple[float, float, float]
    coord_unit: str
    coords: np.ndarray       # (N,3)
    vectors: np.ndarray      # (N,3)
    descriptor: str

def read_fld(path: str | Path) -> FLDData:
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        header1 = f.readline().strip()
        header2 = f.readline().strip()

    m = _HEADER_RE.search(header1)
    if not m:
        raise ValueError(f"Unrecognized HFSS FLD header in {path.name}: {header1}")

    grid_min = _parse_triplet(m.group(1))
    grid_max = _parse_triplet(m.group(2))
    grid_step = _parse_triplet(m.group(3))
    coord_unit = m.group(4)

    try:
        arr = np.loadtxt(path, skiprows=2)
    except ValueError as exc:
        raise ValueError(f"{path.name}: invalid numeric field data: {exc}") from exc
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] != 6:
        raise ValueError(f"{path.name}: expected 6 numeric columns, got {arr.shape[1]}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{path.name}: NaN/Inf detected")
    if not all(np.isfinite(grid_min + grid_max + grid_step)):
        raise ValueError(f"{path.name}: NaN/Inf detected in grid header")

    return FLDData(
        path=path,
        grid_min=grid_min,
        grid_max=grid_max,
        grid_step=grid_step,
        coord_unit=coord_unit,
        coords=arr[:, :3],
        vectors=arr[:, 3:6],
        descriptor=header2,
    )

def pair_real_imag(real_path: str | Path, imag_path: str | Path, atol: float = 1e-12):
    real = read_fld(real_path)
    imag = read_fld(imag_path)

    if not _REAL_DESCRIPTOR_RE.search(real.descriptor):
        raise ValueError(
            f"{real.path.name}: real file descriptor must describe Real(<Ex,Ey,Ez>); "
            f"found {real.descriptor!r}"
        )
    if not _IMAG_DESCRIPTOR_RE.search(imag.descriptor):
        raise ValueError(
            f"{imag.path.name}: imag file descriptor must describe Imag(<Ex,Ey,Ez>); "
            f"found {imag.descriptor!r}"
        )

    if real.coords.shape != imag.coords.shape:
        raise ValueError(f"Grid shape mismatch: {real.path.name} vs {imag.path.name}")
    if not np.allclose(real.coords, imag.coords, rtol=0, atol=atol):
        raise ValueError(f"Coordinate mismatch: {real.path.name} vs {imag.path.name}")
    for attribute in ("grid_min", "grid_max", "grid_step", "coord_unit"):
        if getattr(real, attribute) != getattr(imag, attribute):
            raise ValueError(
                f"Header {attribute} mismatch: {real.path.name} vs {imag.path.name}"
            )

    E = real.vectors.astype(np.float64) + 1j * imag.vectors.astype(np.float64)
    return real.coords, E, real
