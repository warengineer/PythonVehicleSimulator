#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Maneuvering trials of the MSS toolbox (MSS LIBRARY/maneuvering): turning circle, zigzag and pull-out tests.

Each function names its MSS m-file (MSS: github.com/cybergalactic/MSS).
"""

import numpy as np
import math
from .numerics import mround

#------------------------------------------------------------------------------

def turncircle(ship, x, ui, t_final, t_rudderexecute, h):
    """MSS turncircle.m without the plots: Euler integration, rudder command ui from t_rudderexecute (zero
    before). Returns the time series [t u v r x y psi U] (r, psi in rad) and the turning indexes of turncircle.m."""
    x = np.array(x, float)
    N = int(mround(t_final / h))
    out = np.zeros((N + 1, 8))
    store1 = store2 = True
    transfer = advance = tactical = np.nan
    for i in range(N + 1):
        time = i * h
        if mround(abs(x[5]) * 180 / np.pi) >= 90 and store1:
            transfer, advance, store1 = x[4], x[3], False
        if mround(abs(x[5]) * 180 / np.pi) >= 180 and store2:
            tactical, store2 = x[4], False
        if np.ndim(ui):
            u_ship = np.array(ui, float)
            if mround(time) < t_rudderexecute:
                u_ship[0] = 0.0
        else:
            u_ship = 0.0 if mround(time) < t_rudderexecute else ui
        xdot, U = ship(x, u_ship)
        out[i] = np.r_[time, x[0:6], U]
        x = x + h * xdot
    t, u, v, r, xn, ye, psi, U = out.T
    k = int(mround(t_rudderexecute / h)) - 1                # MATLAB index Nrudder (1-based)
    idx = dict(rudder_execute=abs(xn[k]), radius=U[N] / abs(r[N]), max_transfer=np.abs(ye).max(),
               max_advance=abs(np.abs(xn).max() - xn[k]), transfer_90=abs(transfer),
               advance_90=abs(advance - xn[k]), tactical_diameter=abs(tactical))
    return out, idx

#------------------------------------------------------------------------------

def zigzag(ship, x, ui, t_final, t_rudderexecute, h, maneuver=(20.0, 20.0)):
    """MSS zigzag.m without the plots. ui = [delta, ...] (delta = 0 before the execute) or a scalar rudder.
    Returns [t u v r x y psi U delta_c] (angles in rad). The switching logic assumes that a positive rudder turns
    the ship to starboard; call tanker.m through a wrapper that changes the sign of the rudder."""
    x = np.array(x, float)
    N = int(mround(t_final / h))
    out = np.zeros((N + 1, 9))
    u_ship = np.array(ui, float) if np.ndim(ui) else float(ui)

    def setd(d):
        nonlocal u_ship
        if np.ndim(u_ship):
            u_ship[0] = d
        else:
            u_ship = d
    for i in range(N + 1):
        time = i * h
        psi, r = np.rad2deg(x[5]), x[2]
        if mround(time) == t_rudderexecute:
            setd(np.deg2rad(maneuver[0]))
        if mround(time) > t_rudderexecute:
            if psi >= maneuver[1] and r > 0:
                setd(-np.deg2rad(maneuver[0]))
            elif psi <= -maneuver[1] and r < 0:
                setd(np.deg2rad(maneuver[0]))
        xdot, U = ship(x, u_ship)
        d_now = u_ship[0] if np.ndim(u_ship) else u_ship
        out[i] = np.r_[time, x[0:6], U, d_now]
        x = x + h * xdot
    return out
