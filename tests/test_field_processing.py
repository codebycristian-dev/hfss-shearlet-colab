
import numpy as np
import pytest
from src.field_processing import field_magnitude, reshape_plane

def test_field_magnitude_complex():
    E = np.array([[3+4j, 0j, 0j], [0j, 0j, 2j]])
    got = field_magnitude(E)
    assert np.allclose(got, [5., 2.])


def test_reshape_plane_maps_coordinates_not_row_order():
    coords = np.array([
        [1, 7, 20], [0, 7, 10], [0, 7, 20], [1, 7, 10],
    ], dtype=float)
    values = np.array([4, 1, 3, 2], dtype=float)
    x, z, image = reshape_plane(coords, values, "XZ")
    assert np.array_equal(x, [0, 1])
    assert np.array_equal(z, [10, 20])
    assert np.array_equal(image, [[1, 2], [3, 4]])


def test_reshape_plane_rejects_duplicate_or_nonfinite_data():
    coords = np.array([[0, 0, 0], [0, 0, 0]], dtype=float)
    with pytest.raises(ValueError, match="Irregular grid|Duplicate"):
        reshape_plane(coords, np.array([1.0, 2.0]), "XZ")
    with pytest.raises(ValueError, match="NaN/Inf"):
        field_magnitude(np.array([[np.nan + 0j, 0j, 0j]]))
