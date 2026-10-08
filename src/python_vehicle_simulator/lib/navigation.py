#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Signal filters, observers and state estimators of the MSS toolbox (MSS LIBRARY/signalFilters and GNC folders).

Each function names its MSS m-file (MSS: github.com/cybergalactic/MSS).
"""

import numpy as np
import math
from .numerics import rk4

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

#------------------------------------------------------------------------------

def lowPassFilter(xf, u, w_n, h):
    """First-order low-pass filter w_n/(s + w_n), exact discretization (MSS lowPassFilter.m).
    Returns (xf_next, y) with the output y = xf at time k."""
    phi = np.exp(-h * np.asarray(w_n, float))
    return phi * xf + (1 - phi) * u, xf

#------------------------------------------------------------------------------

def highPassFilter(xf, u, w_n, h):
    """First-order high-pass filter s/(s + w_n), exact discretization (MSS highPassFilter.m); y[k] = u[k] - xf[k].
    w_n is a scalar or a vector (the m-file also accepts a diagonal matrix; this port does not)."""
    phi = np.exp(-h * np.asarray(w_n, float))
    y = u - xf
    return phi * xf + (1 - phi) * u, y

#------------------------------------------------------------------------------

def notchFilter(xf, u, w_0, zeta, h, method="RK4"):
    """Second-order notch filter (MSS notchFilter.m).
    'RK4': (s^2 + 2 zeta w_0 s + w_0^2)/(s^2 + 2 w_0 s + w_0^2) in controllable canonical form, one RK4 step with
    u held constant; xf = 2 states. 'IIR': y[n] = u[n] - 2 cos(w_0 h) u[n-1] + u[n-2] + 2 r cos(w_0 h) y[n-1]
    - r^2 y[n-2] with r = exp(-zeta w_0 h); xf = [u[n-1], u[n-2], y[n-1], y[n-2]]."""
    if method == "RK4":
        xf = np.asarray(xf, float).reshape(2)
        A = np.array([[-2 * w_0, -w_0**2], [1.0, 0.0]])
        B = np.array([1.0, 0.0])
        C = np.array([2 * w_0 * (zeta - 1), 0.0])
        y = C @ xf + u
        k1 = A @ xf + B * u
        k2 = A @ (xf + 0.5 * h * k1) + B * u
        k3 = A @ (xf + 0.5 * h * k2) + B * u
        k4 = A @ (xf + h * k3) + B * u
        return xf + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4), y
    if method == "IIR":
        r = math.exp(-zeta * w_0 * h)
        b = [1.0, -2 * math.cos(w_0 * h), 1.0]
        a = [1.0, -2 * r * math.cos(w_0 * h), r * r]
        x_prev, x_prev2, y_prev, y_prev2 = xf
        y = b[0] * u + b[1] * x_prev + b[2] * x_prev2 - a[1] * y_prev - a[2] * y_prev2
        return np.array([u, x_prev, y, y_prev]), y
    raise ValueError("Invalid method. Choose 'RK4' or 'IIR'.")

#------------------------------------------------------------------------------

def waveFreqObserver(x, y, w_f, K_f, h):
    """Nonlinear encounter-frequency observer of Belleter, Galeazzi and Fossen (2015) (MSS waveFreqObserver.m):
    x1' = x2, x2' = w_f^2 (y - x1) - 2 w_f x2, theta' = K_f x1 (x2' - theta x1), omega_e = max(0.4, sqrt|theta|).
    One RK4 step with y held constant. Returns (x_next, omega_e[k])."""
    omega_e = max(0.4, math.sqrt(abs(x[2])))

    def f(z):
        x2dot = w_f**2 * (y - z[0]) - 2 * w_f * z[1]
        return np.array([z[1], x2dot, K_f * z[0] * (x2dot - z[2] * z[0])])
    return rk4(f, h, np.asarray(x, float)), omega_e

#------------------------------------------------------------------------------

def sawToothWave(t):
    """Triangular test signal of period 2 pi between -1 and 1 (MSS sawToothWave.m, width 0.5)."""
    t = np.asarray(t, float)
    width = 0.5
    tn = np.mod(t, 2 * np.pi)
    y = np.zeros_like(t)
    inc = tn < width * 2 * np.pi
    y[inc] = (tn[inc] / (width * 2 * np.pi)) * 2 - 1
    dec = ~inc
    y[dec] = -((tn[dec] - width * 2 * np.pi) / ((1 - width) * 2 * np.pi)) * 2 + 1
    return y
