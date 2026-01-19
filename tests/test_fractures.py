"""
Tests for fractures.py
"""

import numpy as np
from evms.fractures import Fracture, fracture_crossing_weight, compute_D


def test_compute_D():
    D = compute_D(0, 10, 1, 0.5, 0.1)
    assert D == 10 / 1


def test_fracture_crossing():
    frac = Fracture((0,0,0), (0,0,1), (1,0,0), (0,1,0), 2, 2)
    edge = (np.array([-1,0,0]), np.array([1,0,0]))
    w = fracture_crossing_weight(edge, frac)
    assert w >= 0