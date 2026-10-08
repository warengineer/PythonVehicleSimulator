#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thrust configuration and control allocation (MSS thrConfig.m, allocPseudoinverse.m, ...).

Each function names its MSS m-file (MSS: github.com/cybergalactic/MSS).
"""

import numpy as np
import math
from .numerics import invQR

#------------------------------------------------------------------------------

def thrConfig(alpha, l_x, l_y):
    """Thrust configuration matrix T(alpha), 3 x r (MSS thrConfig.m): 'M' main propeller [1, 0, -l_y], 'T' tunnel
    thruster [0, 1, l_x], a number = azimuth angle a: [cos a, sin a, l_x sin a - l_y cos a]."""
    T = np.zeros((3, len(alpha)))
    for i, a in enumerate(alpha):
        if isinstance(a, str) and a == "M":
            T[:, i] = [1.0, 0.0, -l_y[i]]
        elif isinstance(a, str) and a == "T":
            T[:, i] = [0.0, 1.0, l_x[i]]
        else:
            T[:, i] = [np.cos(a), np.sin(a), l_x[i] * np.sin(a) - l_y[i] * np.cos(a)]
    return T

#------------------------------------------------------------------------------

def allocPseudoinverse(K, T, W, tau):
    """Unconstrained allocation u = K^-1 W^-1 T' (T W^-1 T')^-1 tau (MSS allocPseudoinverse.m) for diagonal K, W."""
    K, W = np.asarray(K, float), np.asarray(W, float)
    Winv, Kinv = np.diag(1 / np.diag(W)), np.diag(1 / np.diag(K))
    return Kinv @ Winv @ T.T @ invQR(T @ Winv @ T.T) @ np.asarray(tau, float)
