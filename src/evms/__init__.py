"""
EVMS: Volumetric inversion of geological radioactivity.

Core library for forward modeling and inversion.
"""

from .grid import VoxelGrid
from .forward import build_forward_operator
from .fractures import Fracture, fracture_crossing_weight
from .regularization import build_regularization_matrix
from .inversion import solve_tikhonov, select_lambda
from .calibration import (
    CalibrationModel,
    fit_linear_calibration,
    apply_calibration,
    fit_calibration_from_points,
)
from .io import (
    load_measurements,
    load_grid,
    load_fractures,
    save_grid,
    load_obj_as_grid,
    apply_radioactivity_to_mesh,
    apply_radioactivity_texture,
    export_textured_obj,
)
from .metrics import compute_residuals, compute_correlation
from .metrics import (
    compute_data_misfit_norm,
    compute_regularization_norm,
    compute_holdout_error,
    HoldoutDiagnostics,
)

__all__ = [
    "VoxelGrid",
    "build_forward_operator",
    "Fracture",
    "fracture_crossing_weight",
    "build_regularization_matrix",
    "solve_tikhonov",
    "select_lambda",
    "CalibrationModel",
    "fit_linear_calibration",
    "apply_calibration",
    "fit_calibration_from_points",
    "load_measurements",
    "load_grid",
    "load_fractures",
    "save_grid",
    "load_obj_as_grid",
    "apply_radioactivity_to_mesh",
    "apply_radioactivity_texture",
    "export_textured_obj",
    "compute_residuals",
    "compute_correlation",
    "compute_data_misfit_norm",
    "compute_regularization_norm",
    "compute_holdout_error",
    "HoldoutDiagnostics",
]
