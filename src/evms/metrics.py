"""
Metrics for validation.
"""

import numpy as np
from dataclasses import dataclass


def compute_residuals(A, M: np.ndarray, S_hat: np.ndarray) -> np.ndarray:
    """
    Compute residuals M - A S_hat.
    """
    M_arr = np.asarray(M, dtype=float).ravel()
    S_arr = np.asarray(S_hat, dtype=float).ravel()
    pred = np.asarray(A @ S_arr, dtype=float).ravel()
    if pred.shape != M_arr.shape:
        raise ValueError("A @ S_hat must have the same shape as M")
    if not np.all(np.isfinite(M_arr)):
        raise ValueError("M contains NaN or infinite values")
    if not np.all(np.isfinite(S_arr)):
        raise ValueError("S_hat contains NaN or infinite values")
    return M_arr - pred


def compute_data_misfit_norm(A, M: np.ndarray, S_hat: np.ndarray) -> float:
    """
    Compute ||A S_hat - M||_2.
    """
    res = compute_residuals(A, M, S_hat)
    return float(np.linalg.norm(res))


def compute_regularization_norm(L, S_hat: np.ndarray) -> float:
    """
    Compute ||L S_hat||_2.
    """
    S_arr = np.asarray(S_hat, dtype=float).ravel()
    if not np.all(np.isfinite(S_arr)):
        raise ValueError("S_hat contains NaN or infinite values")
    reg = np.asarray(L @ S_arr, dtype=float).ravel()
    return float(np.linalg.norm(reg))


@dataclass(frozen=True)
class HoldoutDiagnostics:
    """
    Holdout diagnostics computed by refitting on a training subset.
    """

    holdout_rmse: float
    holdout_mae: float
    holdout_l2: float
    holdout_size: int
    train_size: int
    random_state: int
    holdout_fraction: float


def compute_holdout_error(
    A,
    M: np.ndarray,
    L,
    lam: float,
    holdout_fraction: float = 0.2,
    random_state: int = 0,
) -> HoldoutDiagnostics:
    """
    Estimate generalization by fitting on train rows and evaluating on holdout rows.
    """
    if not (0.0 < holdout_fraction < 1.0):
        raise ValueError("holdout_fraction must be in (0, 1)")
    M_arr = np.asarray(M, dtype=float).ravel()
    n = M_arr.size
    if n < 2:
        raise ValueError("At least two measurements are required for holdout diagnostics")
    if not np.all(np.isfinite(M_arr)):
        raise ValueError("M contains NaN or infinite values")

    n_holdout = int(np.floor(holdout_fraction * n))
    n_holdout = max(1, min(n - 1, n_holdout))
    rng = np.random.default_rng(random_state)
    idx = np.arange(n, dtype=int)
    rng.shuffle(idx)
    holdout_idx = idx[:n_holdout]
    train_idx = idx[n_holdout:]

    from .inversion import solve_tikhonov

    A_train = A[train_idx]
    M_train = M_arr[train_idx]
    S_train = solve_tikhonov(A_train, M_train, L, lam)

    A_holdout = A[holdout_idx]
    M_holdout = M_arr[holdout_idx]
    pred_holdout = np.asarray(A_holdout @ S_train, dtype=float).ravel()
    residual = M_holdout - pred_holdout

    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    l2 = float(np.linalg.norm(residual))
    return HoldoutDiagnostics(
        holdout_rmse=rmse,
        holdout_mae=mae,
        holdout_l2=l2,
        holdout_size=int(n_holdout),
        train_size=int(train_idx.size),
        random_state=int(random_state),
        holdout_fraction=float(holdout_fraction),
    )


def compute_correlation(S_true: np.ndarray, S_hat: np.ndarray) -> float:
    """
    Pearson correlation between true and estimated S.
    """
    S_true_arr = np.asarray(S_true, dtype=float).ravel()
    S_hat_arr = np.asarray(S_hat, dtype=float).ravel()
    if S_true_arr.shape != S_hat_arr.shape:
        raise ValueError("S_true and S_hat must have the same shape")
    if not np.all(np.isfinite(S_true_arr)) or not np.all(np.isfinite(S_hat_arr)):
        raise ValueError("S_true and S_hat must not contain NaN/Inf")
    return float(np.corrcoef(S_true_arr, S_hat_arr)[0, 1])
