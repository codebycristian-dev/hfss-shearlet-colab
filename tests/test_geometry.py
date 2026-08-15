
import numpy as np
import pytest
from src.geometry_masks import apply_mask, mask_xz, mask_yz

def test_xz_circle_center_and_edge():
    x = np.array([-50., 0., 50.])
    z = np.array([-50., 0., 50.])
    m = mask_xz(x, z)
    assert m[1,1]
    assert m[1,0] and m[1,2]
    assert not m[0,0]

def test_yz_width_depends_on_x():
    y = np.array([-100., 0., 100.])
    z = np.array([-50., 0., 50.])
    center = mask_yz(y, z, 0.)
    near_edge = mask_yz(y, z, 45.)
    assert center.sum() > near_edge.sum()


def test_mask_is_applied_only_to_copy_and_shape_must_match():
    image = np.ones((2, 2)) * 7
    masked = apply_mask(image, np.array([[True, False], [False, True]]))
    assert np.array_equal(image, np.ones((2, 2)) * 7)
    assert np.array_equal(masked, [[7, 0], [0, 7]])
    with pytest.raises(ValueError, match="shape mismatch"):
        apply_mask(image, np.ones((1, 2), dtype=bool))


def test_yz_rejects_cut_outside_cylinder():
    with pytest.raises(ValueError, match="outside radius"):
        mask_yz(np.array([0.]), np.array([0.]), 51.)
