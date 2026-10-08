#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inertial navigation functions of the MSS toolbox (MSS INS folder): gravity model, attitude observers, aided INS.

Each function names its MSS m-file (MSS: github.com/cybergalactic/MSS).
"""

import numpy as np
import math

#------------------------------------------------------------------------------

def gravity(mu):
    """Acceleration of gravity (m/s^2) at latitude mu (rad), WGS-84 (MSS gravity.m)."""
    s2 = np.sin(mu) ** 2
    return 9.7803253359 * (1 + 0.001931850400 * s2) / np.sqrt(1 - 0.006694384442 * s2)
