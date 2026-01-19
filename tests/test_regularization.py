"""
Tests for regularization.py
"""

import numpy as np
from evms.grid import VoxelGrid
from evms.regularization import build_regularization_matrix


def test_build_reg():
    grid = VoxelGrid((0,0,0), (1,1,1), (2,2,2))
    labels = np.zeros(8)
    L = build_regularization_matrix(grid, labels)
    assert L.shape == (len(grid.neighbor_edges()), 8)


def test_build_reg_respects_layers():
    grid = VoxelGrid((0, 0, 0), (1, 1, 1), (2, 2, 1))
    labels = np.zeros(grid.n_voxels, dtype=int)
    for idx in range(grid.n_voxels):
        i, j, k = grid.flat_to_ijk(idx)
        labels[idx] = (i + j + k) % 2
    L = build_regularization_matrix(grid, labels)
    assert L.nnz == 0
