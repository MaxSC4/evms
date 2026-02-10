"""Tests for calibration.py"""

import numpy as np

from evms.calibration import apply_calibration, fit_calibration_from_points, fit_linear_calibration
from evms.grid import VoxelGrid


def test_fit_linear_calibration_with_offset():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = 2.0 * x + 5.0
    model = fit_linear_calibration(x, y, fit_offset=True)
    assert np.isclose(model.gain, 2.0)
    assert np.isclose(model.offset, 5.0)
    assert np.isclose(model.r2, 1.0)


def test_apply_calibration():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([10.0, 12.0, 14.0])
    model = fit_linear_calibration(x, y, fit_offset=True)
    out = apply_calibration(x, model)
    np.testing.assert_allclose(out, y)


def test_fit_calibration_from_points_nearest_voxel():
    grid = VoxelGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2, 2, 1))
    s_hat = np.array([10.0, 20.0, 30.0, 40.0])
    points = grid.voxel_centers()
    cps = 3.0 * s_hat + 7.0

    model, sampled = fit_calibration_from_points(
        grid=grid,
        s_hat=s_hat,
        calibration_points=points,
        calibration_values=cps,
        fit_offset=True,
    )

    np.testing.assert_allclose(sampled, s_hat)
    assert np.isclose(model.gain, 3.0)
    assert np.isclose(model.offset, 7.0)
