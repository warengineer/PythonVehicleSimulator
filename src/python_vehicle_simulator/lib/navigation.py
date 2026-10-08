#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Signal filters, observers and state estimators of the MSS toolbox (MSS LIBRARY/signalFilters and GNC folders).

Each function names its MSS m-file (MSS: github.com/cybergalactic/MSS).
"""

import numpy as np
import math

#------------------------------------------------------------------------------

class LowPass:
    """First-order low-pass filter object of MSS makeLowPass.m: the call returns the state at time k and then
    propagates x[k+1] = exp(-w_f h) x[k] + (1 - exp(-w_f h)) u[k]."""

    def __init__(self, w_f, h, x0=0.0):
        self.x = x0
        self.phi = np.exp(-h * w_f)

    def __call__(self, u):
        y = self.x
        self.x = self.phi * self.x + (1 - self.phi) * u
        return y
