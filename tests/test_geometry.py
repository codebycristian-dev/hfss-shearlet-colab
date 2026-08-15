
import numpy as np
from src.geometry_masks import mask_xz, mask_yz

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
