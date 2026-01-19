"""
Metrics for validation.
"""

import numpy as np


def compute_residuals(A, M: np.ndarray, S_hat: np.ndarray) -> np.ndarray:
    """
    Compute residuals M - A S_hat.
    """
    return M - A @ S_hat


def compute_correlation(S_true: np.ndarray, S_hat: np.ndarray) -> float:
    """
    Pearson correlation between true and estimated S.
    """
    return np.corrcoef(S_true, S_hat)[0,1]