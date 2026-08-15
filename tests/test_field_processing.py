
import numpy as np
from src.field_processing import field_magnitude

def test_field_magnitude_complex():
    E = np.array([[3+4j, 0j, 0j], [0j, 0j, 2j]])
    got = field_magnitude(E)
    assert np.allclose(got, [5., 2.])
