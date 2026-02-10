"""
Calibration utilities for mapping relative inverted source intensity to physical units.

Model:
    y_phys = gain * x_rel + offset
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree

from .grid import VoxelGrid


@dataclass(frozen=True)
class CalibrationModel:
    """Linear calibration model from relative to physical units."""

    gain: float
    offset: float
    r2: float
    n_samples: int


def _validate_1d(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def fit_linear_calibration(
    relative_values: np.ndarray,
    physical_values: np.ndarray,
    fit_offset: bool = True,
) -> CalibrationModel:
    """
    Fit a linear mapping from relative values to physical values.

    Args:
        relative_values: 1D array in relative units.
        physical_values: 1D array in physical units (e.g., cps/s).
        fit_offset: If True fit y = gain*x + offset, else y = gain*x.

    Returns:
        CalibrationModel with gain, offset and R^2.
    """
    x = _validate_1d("relative_values", relative_values)
    y = _validate_1d("physical_values", physical_values)
    if x.shape != y.shape:
        raise ValueError("relative_values and physical_values must have the same shape")
    if x.size < 2:
        raise ValueError("at least two samples are required for calibration")

    if fit_offset:
        design = np.column_stack((x, np.ones_like(x)))
        coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        gain, offset = float(coeffs[0]), float(coeffs[1])
    else:
        denom = float(np.dot(x, x))
        if denom <= 0.0:
            raise ValueError("cannot fit gain-only model when relative_values are all zero")
        gain = float(np.dot(x, y) / denom)
        offset = 0.0

    y_pred = gain * x + offset
    residual = y - y_pred
    ss_res = float(np.dot(residual, residual))
    y_mean = float(np.mean(y))
    centered = y - y_mean
    ss_tot = float(np.dot(centered, centered))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - (ss_res / ss_tot)

    return CalibrationModel(gain=gain, offset=offset, r2=r2, n_samples=int(x.size))


def apply_calibration(values: np.ndarray, model: CalibrationModel) -> np.ndarray:
    """Apply a fitted linear calibration model."""
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("values contains NaN or infinite values")
    return model.gain * arr + model.offset


def fit_calibration_from_points(
    grid: VoxelGrid,
    s_hat: np.ndarray,
    calibration_points: np.ndarray,
    calibration_values: np.ndarray,
    fit_offset: bool = True,
) -> Tuple[CalibrationModel, np.ndarray]:
    """
    Fit calibration from point observations against nearest reconstructed voxels.

    Args:
        grid: Voxel grid used for inversion.
        s_hat: Reconstructed source intensity per active voxel.
        calibration_points: Array (n,3) of point coordinates.
        calibration_values: Array (n,) observed physical values at those points.
        fit_offset: If True fit y = gain*x + offset.

    Returns:
        model: CalibrationModel
        matched_relative_values: Relative values sampled from nearest voxels.
    """
    s = _validate_1d("s_hat", s_hat)
    if s.size != grid.n_voxels:
        raise ValueError("s_hat size must match number of active voxels")

    points = np.asarray(calibration_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("calibration_points must have shape (n_points, 3)")
    if points.shape[0] == 0:
        raise ValueError("calibration_points must not be empty")
    if not np.all(np.isfinite(points)):
        raise ValueError("calibration_points contains NaN or infinite values")

    y = _validate_1d("calibration_values", calibration_values)
    if y.size != points.shape[0]:
        raise ValueError("calibration_values length must match number of calibration_points")

    centers = grid.voxel_centers()
    tree = cKDTree(centers)
    _, idx = tree.query(points)
    x = s[idx]

    model = fit_linear_calibration(x, y, fit_offset=fit_offset)
    return model, x
