"""
Inversion solvers and lambda selection.

Solve min ||A S - M||^2 + λ ||L S||^2 using LSMR or CG.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr, LinearOperator
from typing import Tuple


def solve_tikhonov(A, M: np.ndarray, L, lam: float) -> np.ndarray:
    """
    Solve Tikhonov regularization.

    Args:
        A: Forward operator (sparse or LinearOperator).
        M: Measurements vector.
        L: Regularization matrix (sparse).
        lam: Regularization parameter.

    Returns:
        S_hat: Estimated S.
    """
    # Augment system: [A; sqrt(lam) L] S = [M; 0]
    sqrt_lam = np.sqrt(lam)
    A_aug = sparse.vstack([A, sqrt_lam * L])
    M_aug = np.concatenate([M, np.zeros(L.shape[0])])

    # Solve using LSQR
    S_hat, _, _, _, _, _, _ = lsqr(A_aug, M_aug)[:7]
    return S_hat


def select_lambda(A, M: np.ndarray, L, lambda_grid: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Select lambda using L-curve heuristic.

    Compute curvature of L-curve and pick max.

    Args:
        A, M, L: As above.
        lambda_grid: Array of lambda values.

    Returns:
        Best lambda, array of (residual, reg_norm) for each lambda.
    """
    res_norms = []
    reg_norms = []
    for lam in lambda_grid:
        S_hat = solve_tikhonov(A, M, L, lam)
        res = np.linalg.norm(A @ S_hat - M)
        reg = np.linalg.norm(L @ S_hat)
        res_norms.append(res)
        reg_norms.append(reg)

    res_norms = np.array(res_norms)
    reg_norms = np.array(reg_norms)

    # L-curve curvature (simple approximation)
    log_res = np.log(res_norms)
    log_reg = np.log(reg_norms)
    curv = np.abs(np.gradient(np.gradient(log_res, log_reg), log_reg))
    best_idx = np.argmax(curv)
    return lambda_grid[best_idx], np.column_stack((res_norms, reg_norms))