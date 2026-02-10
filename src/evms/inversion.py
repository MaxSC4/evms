"""
Inversion solvers and lambda selection.

Solve min ||A S - M||^2 + λ ||L S||^2 using LSMR or CG.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr, LinearOperator
from typing import Tuple

from .forward import build_forward_operator


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


def select_forward_params(
    grid,
    measurement_points: np.ndarray,
    M: np.ndarray,
    L,
    mu_grid: np.ndarray,
    rmax_grid: np.ndarray,
    lam: float = None,
    lambda_grid: np.ndarray = None,
    objective: str = "residual",
    holdout_fraction: float = 0.2,
    random_state: int = 0,
) -> Tuple[float, float, float, np.ndarray]:
    """
    Grid-search mu and R_max by minimizing a scalar objective.

    Args:
        grid: VoxelGrid used for forward operator construction.
        measurement_points: Array of shape (n_points, 3).
        M: Measurements vector.
        L: Regularization matrix.
        mu_grid: Candidate attenuation coefficients.
        rmax_grid: Candidate influence radii.
        lam: Fixed lambda (used if lambda_grid is None).
        lambda_grid: Optional lambda candidates for per-candidate L-curve selection.
        objective: "residual" for ||A S - M||, "holdout" for holdout RMSE.
        holdout_fraction: Holdout fraction when objective="holdout".
        random_state: Seed for holdout split when objective="holdout".

    Returns:
        best_mu, best_rmax, best_lam, table
        where table columns are [mu, rmax, lam, score].
    """
    mu_vals = np.asarray(mu_grid, dtype=float).ravel()
    rmax_vals = np.asarray(rmax_grid, dtype=float).ravel()
    if mu_vals.size == 0 or rmax_vals.size == 0:
        raise ValueError("mu_grid and rmax_grid must not be empty")
    if not np.all(np.isfinite(mu_vals)) or not np.all(np.isfinite(rmax_vals)):
        raise ValueError("mu_grid and rmax_grid must contain finite values")
    if np.any(mu_vals < 0.0):
        raise ValueError("mu_grid must contain non-negative values")
    if np.any(rmax_vals <= 0.0):
        raise ValueError("rmax_grid must contain positive values")
    if objective not in {"residual", "holdout"}:
        raise ValueError("objective must be 'residual' or 'holdout'")
    if lambda_grid is None and lam is None:
        raise ValueError("Provide either lam or lambda_grid")

    M_arr = np.asarray(M, dtype=float).ravel()
    if not np.all(np.isfinite(M_arr)):
        raise ValueError("M contains NaN or infinite values")

    records = []
    best = None
    for mu in mu_vals:
        for rmax in rmax_vals:
            A = build_forward_operator(grid, measurement_points, float(mu), float(rmax))
            if lambda_grid is not None:
                lam_i, _ = select_lambda(A, M_arr, L, np.asarray(lambda_grid, dtype=float))
            else:
                lam_i = float(lam)

            S_hat = solve_tikhonov(A, M_arr, L, lam_i)
            if objective == "residual":
                score = float(np.linalg.norm(A @ S_hat - M_arr))
            else:
                from .metrics import compute_holdout_error

                holdout_diag = compute_holdout_error(
                    A=A,
                    M=M_arr,
                    L=L,
                    lam=lam_i,
                    holdout_fraction=holdout_fraction,
                    random_state=random_state,
                )
                score = float(holdout_diag.holdout_rmse)

            rec = (float(mu), float(rmax), float(lam_i), score)
            records.append(rec)
            if best is None or score < best[3]:
                best = rec

    table = np.asarray(records, dtype=float)
    return best[0], best[1], best[2], table
