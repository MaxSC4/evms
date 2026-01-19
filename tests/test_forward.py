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


def test_build_forward_matches_bruteforce():
    grid = VoxelGrid((0, 0, 0), (1, 1, 1), (2, 2, 2))
    points = np.array([[0.5, 0.5, 0.5], [3.0, 3.0, 3.0]])
    mu = 0.02
    R_max = 2.0
    eps = 1e-6

    A = build_forward_operator(grid, points, mu, R_max, eps).toarray()
    centers = grid.voxel_centers()
    volume = np.prod(grid.spacing)
    expected = np.zeros_like(A)
    for i, x in enumerate(points):
        for j, r in enumerate(centers):
            dist = np.linalg.norm(x - r)
            if dist < R_max:
                g = 1 / (dist**2 + eps)
                atten = np.exp(-mu * dist)
                expected[i, j] = g * atten * volume

    np.testing.assert_allclose(A, expected, rtol=1e-10, atol=1e-12)
