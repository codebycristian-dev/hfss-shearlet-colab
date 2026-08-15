
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

EXPECTED_XZ = tuple(range(-90, 91, 20))
EXPECTED_YZ = tuple(range(-45, 46, 10))
POLARIZATION_BY_MODE = {
    1: "parallel_y",
    2: "perpendicular_y",
}
TARGET_DIRECTORIES = {
    (1, "XZ"): Path("01_Phantom_Mode1/01_transverse_XZ"),
    (1, "YZ"): Path("01_Phantom_Mode1/02_axial_YZ"),
    (2, "XZ"): Path("02_Phantom_Mode2/01_transverse_XZ"),
    (2, "YZ"): Path("02_Phantom_Mode2/02_axial_YZ"),
}

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
    if not data_root.is_dir():
        raise FileNotFoundError(f"HFSS data directory not found: {data_root}")
    slots: dict[tuple, dict[str, Path]] = {}
    target_paths: list[tuple[int, str, Path]] = []
    for (expected_mode, expected_plane), relative in TARGET_DIRECTORIES.items():
        matches = [p for p in data_root.rglob(relative.name) if p.is_dir() and p.parts[-2:] == relative.parts[-2:]]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one target directory ending in {relative}, found {len(matches)}"
            )
        target_paths.append((expected_mode, expected_plane, matches[0]))

    for expected_mode, expected_plane, target in target_paths:
        for p in target.rglob("*.fld"):
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
                raise ValueError(f"Unrecognized HFSS .fld filename inside target directory: {p}")
            if key[:2] != (expected_mode, expected_plane):
                raise ValueError(
                    f"Field {p.name} declares Mode{key[0]} {key[1]} but is inside "
                    f"Mode{expected_mode} {expected_plane} target directory"
                )
            slot = slots.setdefault(key, {})
            if part in slot:
                raise ValueError(
                    f"Duplicate {part} field for {key}: {slot[part]} and {p}"
                )
            slot[part] = p

    pairs = []
    for (mode, plane, idx, fixed), d in sorted(slots.items()):
        if "real" not in d or "imag" not in d:
            raise ValueError(f"Missing real/imag pair for {(mode, plane, idx, fixed)}")
        pairs.append(FieldPair(mode, plane, idx, fixed, d["real"], d["imag"]))
    return pairs

def validate_expected_40(pairs: list[FieldPair]) -> None:
    if len(pairs) != 40:
        raise ValueError(f"Expected exactly 40 phantom pairs, found {len(pairs)}")

    for mode in (1, 2):
        xz = [p for p in pairs if p.mode == mode and p.plane == "XZ"]
        yz = [p for p in pairs if p.mode == mode and p.plane == "YZ"]
        expected_xz = {(i, fixed) for i, fixed in enumerate(EXPECTED_XZ, 1)}
        expected_yz = {(i, fixed) for i, fixed in enumerate(EXPECTED_YZ, 1)}
        got_xz = {(p.index, p.fixed_mm) for p in xz}
        got_yz = {(p.index, p.fixed_mm) for p in yz}
        if got_xz != expected_xz or len(xz) != 10:
            raise ValueError(
                f"Mode {mode}: invalid XZ cuts; expected {sorted(expected_xz)}, "
                f"found {sorted(got_xz)}"
            )
        if got_yz != expected_yz or len(yz) != 10:
            raise ValueError(
                f"Mode {mode}: invalid YZ cuts; expected {sorted(expected_yz)}, "
                f"found {sorted(got_yz)}"
            )
