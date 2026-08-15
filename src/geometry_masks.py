
from __future__ import annotations
import numpy as np

R_PH_MM = 50.0
HALF_LENGTH_Y_MM = 100.0

def mask_xz(x_mm: np.ndarray, z_mm: np.ndarray, radius_mm: float = R_PH_MM):
    X, Z = np.meshgrid(x_mm, z_mm)
    return (X**2 + Z**2) <= radius_mm**2 + 1e-12

def mask_yz(y_mm: np.ndarray, z_mm: np.ndarray, fixed_x_mm: float,
            radius_mm: float = R_PH_MM, half_length_y_mm: float = HALF_LENGTH_Y_MM):
    z_half = np.sqrt(max(radius_mm**2 - fixed_x_mm**2, 0.0))
    Y, Z = np.meshgrid(y_mm, z_mm)
    return (np.abs(Y) <= half_length_y_mm + 1e-12) & (np.abs(Z) <= z_half + 1e-12)

def apply_mask(img, mask, outside_value=0.0):
    out = np.array(img, dtype=float, copy=True)
    out[~mask] = outside_value
    return out
