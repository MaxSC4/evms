"""
Fracture utilities for finite barrier patches.

Handles detection of edge crossings and weight computation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Fracture:
    """
    Finite rectangular fracture patch.

    Attributes:
        center: (px, py, pz) center point.
        normal: (nx, ny, nz) unit normal.
        u: (ux, uy, uz) in-plane direction (length).
        v: (vx, vy, vz) in-plane direction (width).
        length: L along u.
        width: W along v.
        rho0: Base ratio.
        rho1: Slope for linear ratio.
        rho_min: Min clamp for rho.
        alpha: Weight multiplier.
    """
    center: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    u: Tuple[float, float, float]
    v: Tuple[float, float, float]
    length: float
    width: float
    rho0: float = 1.0
    rho1: float = 0.5
    rho_min: float = 0.1
    alpha: float = 1.0


def segment_plane_intersection(r1: np.ndarray, r2: np.ndarray, p: np.ndarray, n: np.ndarray) -> Tuple[bool, np.ndarray]:
    """
    Check if segment [r1, r2] crosses plane at p with normal n.

    Returns:
        (crosses, r_star) where r_star is intersection point if crosses.
    """
    d1 = np.dot(n, r1 - p)
    d2 = np.dot(n, r2 - p)
    if d1 * d2 >= 0:
        return False, np.zeros(3)
    # Interpolate
    t = d1 / (d1 - d2)
    r_star = r1 + t * (r2 - r1)
    return True, r_star


def point_in_rectangle(r_star: np.ndarray, p: np.ndarray, u: np.ndarray, v: np.ndarray, L: float, W: float) -> bool:
    """Check if r_star is inside rectangle."""
    vec = r_star - p
    s_u = np.dot(vec, u)
    s_v = np.dot(vec, v)
    return abs(s_u) <= L/2 and abs(s_v) <= W/2


def compute_D(s: float, L: float, rho0: float, rho1: float, rho_min: float) -> float:
    """Compute D(s) from linear ratio law."""
    ratio = rho0 + rho1 * abs(s) / (L/2)
    rho = max(ratio, rho_min)
    return L / rho


def fracture_crossing_weight(edge: Tuple[np.ndarray, np.ndarray], fracture: Fracture) -> float:
    """
    Compute w_fract contribution for an edge if it crosses the fracture.

    Args:
        edge: (r_j, r_j') two points.
        fracture: Fracture object.

    Returns:
        alpha * g(D) if crosses, else 0. g(D) = D / D_ref with D_ref = L.
    """
    r1, r2 = edge
    p = np.array(fracture.center)
    n = np.array(fracture.normal)
    crosses, r_star = segment_plane_intersection(r1, r2, p, n)
    if not crosses:
        return 0.0
    u = np.array(fracture.u)
    v = np.array(fracture.v)
    if not point_in_rectangle(r_star, p, u, v, fracture.length, fracture.width):
        return 0.0
    vec = r_star - p
    s = np.dot(vec, u)
    D = compute_D(s, fracture.length, fracture.rho0, fracture.rho1, fracture.rho_min)
    D_ref = fracture.length
    g = D / D_ref
    return fracture.alpha * g