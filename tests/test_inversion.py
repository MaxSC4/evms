"""
Tests for inversion.py
"""

import numpy as np
from scipy import sparse
from evms.grid import VoxelGrid
from evms.forward import build_forward_operator
from evms.inversion import solve_tikhonov, select_lambda, select_forward_params


def test_solve_tikhonov():
    A = sparse.csr_matrix(np.eye(5))
    M = np.ones(5)
    L = sparse.csr_matrix(np.eye(5))
    S = solve_tikhonov(A, M, L, 0.1)
    np.testing.assert_array_almost_equal(S, M, decimal=1)


def test_select_lambda():
    A = sparse.csr_matrix(np.eye(5))
    M = np.ones(5)
    L = sparse.csr_matrix(np.eye(5))
    lam, _ = select_lambda(A, M, L, np.array([0.1, 1.0]))
    assert lam in [0.1, 1.0]


def test_select_forward_params_residual():
    grid = VoxelGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2, 2, 1))
    points = grid.voxel_centers()
    s_true = np.array([1.0, 2.0, 3.0, 4.0])
    mu_true = 0.01
    rmax_true = 3.0
    A_true = build_forward_operator(grid, points, mu_true, rmax_true)
    M = A_true @ s_true
    L = sparse.csr_matrix(np.eye(grid.n_voxels))

    mu, rmax, lam, table = select_forward_params(
        grid=grid,
        measurement_points=points,
        M=M,
        L=L,
        mu_grid=np.array([0.0, mu_true, 0.03]),
        rmax_grid=np.array([1.5, rmax_true]),
        lam=1e-8,
        objective="residual",
    )

    assert mu in [0.0, mu_true, 0.03]
    assert rmax in [1.5, rmax_true]
    assert np.isclose(lam, 1e-8)
    assert table.shape == (6, 4)
    assert np.isclose(table[:, 3].min(), table[(table[:, 0] == mu) & (table[:, 1] == rmax), 3][0])


def test_select_forward_params_holdout():
    grid = VoxelGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2, 2, 2))
    points = grid.voxel_centers()
    s_true = np.linspace(1.0, 8.0, grid.n_voxels)
    A_true = build_forward_operator(grid, points, 0.02, 3.0)
    M = A_true @ s_true
    L = sparse.csr_matrix(np.eye(grid.n_voxels))

    mu, rmax, lam, table = select_forward_params(
        grid=grid,
        measurement_points=points,
        M=M,
        L=L,
        mu_grid=np.array([0.0, 0.02]),
        rmax_grid=np.array([2.0, 3.0]),
        lam=1e-4,
        objective="holdout",
        holdout_fraction=0.25,
        random_state=0,
    )

    assert mu in [0.0, 0.02]
    assert rmax in [2.0, 3.0]
    assert np.isclose(lam, 1e-4)
    assert table.shape == (4, 4)
