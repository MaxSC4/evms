"""
Tests for forward.py
"""

import numpy as np
from evms.grid import VoxelGrid
from evms.forward import build_forward_operator


def test_build_forward():
    grid = VoxelGrid((0,0,0), (1,1,1), (2,2,2))
    points = np.array([[0.5,0.5,0.5]])
    A = build_forward_operator(grid, points, 0.01, 10.0)
    assert A.shape == (1, 8)
    assert A.nnz > 0