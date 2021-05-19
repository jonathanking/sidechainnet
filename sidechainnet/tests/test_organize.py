import numpy as np
import pytest
from sidechainnet.utils.organize import compute_angle_means


def test_compute_angle_means():
    # Zeros are considered padded
    angles = [
        np.asarray([[1.] * 12]*20),
        np.asarray([[0.] * 12]*20),
        np.asarray([[0.] * 12]*20),
    ]
    means = compute_angle_means(angles)
    np.testing.assert_array_almost_equal(means, np.asarray([1.] * 12))

    angles = [
        np.asarray([[1.] * 12] * 20),
        np.asarray([[3.] * 12] * 20),
        np.asarray([[0.] * 12] * 20),
    ]
    means = compute_angle_means(angles)
    np.testing.assert_array_almost_equal(means, np.asarray([2.] * 12))

    angles = [
        np.asarray([[.2] * 10 + [0, 0]] * 20),
        np.asarray([[.4] * 10 + [0, 0]] * 20),
        np.asarray([[0.] * 12] * 20),
    ]
    means = compute_angle_means(angles)
    np.testing.assert_array_almost_equal(means, np.asarray([.3] * 10 + [0, 0]))
