"""
Tests for inversion.py
"""

import numpy as np
from scipy import sparse
from evms.inversion import solve_tikhonov, select_lambda


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