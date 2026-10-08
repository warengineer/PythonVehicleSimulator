#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The pytest for 'test_crossflow_drag.py' can be run in the terminal using:
1) cd <path of the Python Vehicle Simulator installation>
2) pytest -k crossflow -v
"""

import numpy as np
import math
from python_vehicle_simulator.lib.gnc import crossFlowDrag, Hoerner

RHO = 1025  # water density used internally by crossFlowDrag (as in MSS)


def test_pure_sway_force_matches_analytic_integral():
    """In pure sway (v != 0, r = 0) the cross-flow term |v + x*r|(v + x*r)
    reduces to the constant |v|v, so the strip integral has the closed form
    Y = -1/2 * rho * T * Cd_2D * |v|v * L. A quadrature spanning exactly
    [-L/2, L/2] reproduces this to machine precision."""
    L, B, T = 1.6, 0.19, 0.19
    v = 0.8
    nu_r = np.array([0, v, 0, 0, 0, 0], float)

    Y_true = -0.5 * RHO * T * Hoerner(B, T) * abs(v) * v * L
    tau = crossFlowDrag(L, B, T, nu_r)

    assert math.isclose(tau[1], Y_true, rel_tol=1e-9)


def test_pure_yaw_moment_matches_analytic_integral():
    """In pure yaw (v = 0, r != 0) the sway integrand |x*r|(x*r) is odd in x,
    so the net sway force vanishes by symmetry, while the moment integrand
    x * |x*r|(x*r) = |r|r * |x|^3 gives the closed form
    N = -1/2 * rho * T * Cd_2D * |r|r * integral(|x|^3) = -... * L^4/32.
    The 1% tolerance covers the midpoint-rule discretization error
    (about 0.5% at n = 20, dominated by the |x|^3 kink at x = 0)."""
    L, B, T = 1.6, 0.19, 0.19
    r = 0.5
    nu_r = np.array([0, 0, 0, 0, 0, r], float)

    N_true = -0.5 * RHO * T * Hoerner(B, T) * abs(r) * r * L**4 / 32
    tau = crossFlowDrag(L, B, T, nu_r)

    assert abs(tau[1]) < 1e-9
    assert math.isclose(tau[5], N_true, rel_tol=0.01)
