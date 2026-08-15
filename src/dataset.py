
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re

_TRANSVERSE_RE = re.compile(
    r"E_(real|imag)_T(\d{2})_XZ_y([+-]\d{3})_M([12])_2p45GHz\.fld$"
)
_AXIAL_RE = re.compile(
    r"E_(real|imag)_A(\d{2})_YZ_x([+-]\d{3})_M([12])_2p45GHz\.fld$"
)

@dataclass(frozen=True)
class FieldPair:
    mode: int
    plane: str
    index: int
    fixed_mm: int
    real_path: Path
    imag_path: Path

def _signed_int(s: str) -> int:
    return int(s)

def discover_phantom_pairs(data_root: str | Path) -> list[FieldPair]:
    data_root = Path(data_root)
    slots: dict[tuple, dict[str, Path]] = {}

    for p in data_root.rglob("*.fld"):
        name = p.name
        mt = _TRANSVERSE_RE.match(name)
        ma = _AXIAL_RE.match(name)
        if mt:
            part, idx, fixed, mode = mt.groups()
            key = (int(mode), "XZ", int(idx), _signed_int(fixed))
        elif ma:
            part, idx, fixed, mode = ma.groups()
            key = (int(mode), "YZ", int(idx), _signed_int(fixed))
        else:
            continue
        slots.setdefault(key, {})[part] = p

    pairs = []
    for (mode, plane, idx, fixed), d in sorted(slots.items()):
        if "real" not in d or "imag" not in d:
            raise ValueError(f"Missing real/imag pair for {(mode, plane, idx, fixed)}")
        pairs.append(FieldPair(mode, plane, idx, fixed, d["real"], d["imag"]))
    return pairs

def validate_expected_40(pairs: list[FieldPair]) -> None:
    expected_xz = {-90, -70, -50, -30, -10, 10, 30, 50, 70, 90}
    expected_yz = {-45, -35, -25, -15, -5, 5, 15, 25, 35, 45}

    for mode in (1, 2):
        xz = [p for p in pairs if p.mode == mode and p.plane == "XZ"]
        yz = [p for p in pairs if p.mode == mode and p.plane == "YZ"]
        if len(xz) != 10 or {p.fixed_mm for p in xz} != expected_xz:
            raise ValueError(f"Mode {mode}: invalid XZ set ({len(xz)} cuts)")
        if len(yz) != 10 or {p.fixed_mm for p in yz} != expected_yz:
            raise ValueError(f"Mode {mode}: invalid YZ set ({len(yz)} cuts)")
    if len(pairs) != 40:
        raise ValueError(f"Expected 40 phantom pairs, found {len(pairs)}")
