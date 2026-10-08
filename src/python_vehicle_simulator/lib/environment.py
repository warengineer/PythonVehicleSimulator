#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environmental models of the MSS toolbox (MSS LIBRARY/environment and LIBRARY/motionSickness): wave spectra,
encounter frequency, wind and current loads, motion sickness incidence.

Each function names its MSS m-file (MSS: github.com/cybergalactic/MSS).
"""

import numpy as np
import math
from .gnc import G_MSS

#------------------------------------------------------------------------------

def encounter(chi, U, Omega):
    """Encounter frequency w_e = w - w^2 U cos(chi) / g (MSS encounter.m; chi = 0 following sea, pi head sea)."""
    return Omega - (np.asarray(Omega)**2 * U * np.cos(chi) / G_MSS)

#------------------------------------------------------------------------------

def ISOmsi(t):
    """ISO 2631 MSI of 10 % after t hours (MSS ISOmsi.m): f = 0.1:0.03:0.63 Hz, a_z = 0.5 sqrt(2/t) for f < 0.315 Hz
    and 0.5 sqrt(2/t) 6.8837 f^1.67 for f >= 0.315 Hz. Returns (a_z in m/s^2, w_e = 2 pi f in rad/s)."""
    f = 0.1 + 0.03 * np.arange(18)
    t = max(t, np.finfo(float).eps)
    a_z = 0.5 * np.sqrt(2 / t) * np.ones_like(f)
    hi = f >= 0.315
    a_z[hi] = a_z[hi] * (6.8837 * f[hi]**1.67)
    return a_z, 2 * np.pi * f

#------------------------------------------------------------------------------

def _erf(x):
    return np.vectorize(math.erf)(x)

#------------------------------------------------------------------------------

def HMmsi(a_z, w_e):
    """O'Hanlon and McCauley (1974) MSI in percent after two hours (MSS HMmsi.m, revision 2025-02-25): a_z is
    replaced by its rms value over the vector, f_e = w_e/(2 pi) in Hz, mu = -0.819 + 2.32 (log10 f_e)^2,
    MSI = 50 (1 - erf((-log10(a_z,rms/g) + mu)/(0.4 sqrt(2))))."""
    a_z = np.atleast_1d(np.asarray(a_z, float)).ravel()
    w_e = np.atleast_1d(np.asarray(w_e, float)).ravel()
    a_rms = np.sqrt(np.mean(a_z**2))
    f_e = w_e / (2 * np.pi)
    mu = -0.819 + 2.32 * np.log10(np.maximum(f_e, 1e-6))**2
    I = (-np.log10(a_rms / 9.81) + mu) / (0.4 * np.sqrt(2))
    return 100 * (0.5 * (1 - _erf(I)))
