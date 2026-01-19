"""
EVMS: Volumetric inversion of geological radioactivity.

Core library for forward modeling and inversion.
"""

from .grid import VoxelGrid
from .forward import build_forward_operator
from .fractures import Fracture, fracture_crossing_weight
from .regularization import build_regularization_matrix
from .inversion import solve_tikhonov, select_lambda
from .io import (
    load_measurements,
    load_grid,
    load_fractures,
    save_grid,
    load_obj_as_grid,
    apply_radioactivity_to_mesh,
    apply_radioactivity_texture,
)
from .metrics import compute_residuals, compute_correlation

__all__ = [
    "VoxelGrid",
    "build_forward_operator",
    "Fracture",
    "fracture_crossing_weight",
    "build_regularization_matrix",
    "solve_tikhonov",
    "select_lambda",
    "load_measurements",
    "load_grid",
    "load_fractures",
    "save_grid",
    "load_obj_as_grid",
    "apply_radioactivity_to_mesh",
    "apply_radioactivity_texture",
    "compute_residuals",
    "compute_correlation",
]
