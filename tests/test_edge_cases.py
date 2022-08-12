import numpy as np
import pytest

from gaussian_mixture.check_input import check_input


def test_check_input_rejects_nan():
    with pytest.raises(ValueError):
        check_input(np.array([[1.0], [np.nan]]))


def test_check_input_rejects_one_dimensional_input():
    with pytest.raises(ValueError):
        check_input(np.array([1.0, 2.0, 3.0]))
