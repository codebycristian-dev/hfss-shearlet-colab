
from __future__ import annotations
import numpy as np

def field_magnitude(E: np.ndarray) -> np.ndarray:
    """|E| = sqrt(|Ex|^2 + |Ey|^2 + |Ez|^2), in V/m."""
    if E.ndim != 2 or E.shape[1] != 3:
        raise ValueError("E must have shape (N,3)")
    if not np.isfinite(E).all():
        raise ValueError("E contains NaN/Inf")
    return np.sqrt(np.sum(np.abs(E) ** 2, axis=1))

def reshape_plane(coords: np.ndarray, values: np.ndarray, plane: str):
    """Return axis1, axis2, image with physical ordering preserved."""
    coords = np.asarray(coords)
    values = np.asarray(values)
    if coords.ndim != 2 or coords.shape[1] != 3 or values.ndim != 1:
        raise ValueError("coords must have shape (N,3) and values shape (N,)")
    if len(coords) != len(values):
        raise ValueError("Coordinate/value count mismatch")
    if not np.isfinite(coords).all() or not np.isfinite(values).all():
        raise ValueError("Coordinates or values contain NaN/Inf")
    plane = plane.upper()
    if plane == "XZ":
        a1, a2 = coords[:, 0], coords[:, 2]
    elif plane == "YZ":
        a1, a2 = coords[:, 1], coords[:, 2]
    else:
        raise ValueError(f"Unsupported plane {plane!r}")

    u1 = np.unique(a1)
    u2 = np.unique(a2)
    expected = len(u1) * len(u2)
    if len(values) != expected:
        raise ValueError(f"Irregular grid: {len(values)} values vs {expected} expected")

    # Do not assume HFSS serialization order.
    i1 = {v: i for i, v in enumerate(u1)}
    i2 = {v: i for i, v in enumerate(u2)}
    img = np.full((len(u2), len(u1)), np.nan, dtype=float)
    for x1, x2, val in zip(a1, a2, values):
        if np.isfinite(img[i2[x2], i1[x1]]):
            raise ValueError(f"Duplicate grid coordinate ({x1}, {x2})")
        img[i2[x2], i1[x1]] = val

    if not np.isfinite(img).all():
        raise ValueError("Incomplete grid after reshape")
    return u1, u2, img
