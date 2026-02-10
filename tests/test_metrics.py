"""Tests for metrics.py"""

import numpy as np
from scipy import sparse

from evms.metrics import (
    compute_residuals,
    compute_data_misfit_norm,
    compute_regularization_norm,
    compute_holdout_error,
)


def test_data_misfit_norm_matches_residual_norm():
    A = sparse.csr_matrix(np.eye(3))
    M = np.array([1.0, 2.0, 3.0])
    S = np.array([0.5, 2.5, 2.0])
    res = compute_residuals(A, M, S)
    misfit = compute_data_misfit_norm(A, M, S)
    assert np.isclose(misfit, np.linalg.norm(res))


def test_regularization_norm_identity():
    L = sparse.csr_matrix(np.eye(4))
    S = np.array([1.0, -2.0, 3.0, -4.0])
    reg = compute_regularization_norm(L, S)
    assert np.isclose(reg, np.linalg.norm(S))


def test_holdout_error_runs_and_returns_sizes():
    A = sparse.csr_matrix(np.eye(10))
    M = np.linspace(1.0, 10.0, 10)
    L = sparse.csr_matrix(np.eye(10))

    diag = compute_holdout_error(A, M, L, lam=0.1, holdout_fraction=0.2, random_state=0)

    assert diag.holdout_size == 2
    assert diag.train_size == 8
    assert diag.holdout_rmse >= 0.0
    assert diag.holdout_mae >= 0.0
    assert diag.holdout_l2 >= 0.0
