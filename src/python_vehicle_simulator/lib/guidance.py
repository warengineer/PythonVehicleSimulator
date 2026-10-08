#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guidance algorithms.

Reference: T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and
Motion Control. 2nd. Edition, Wiley. 
URL: www.fossen.biz/wiley

Author:     Thor I. Fossen
"""

import numpy as np
import math
from .gnc import ssa

# [x_d,v_d,a_d] = refModel3(x_d,v_d,a_d,r,wn_d,zeta_d,v_max,sampleTime) is a 3-order 
# reference  model for generation of a smooth desired position x_d, velocity |v_d| < v_max, 
# and acceleration a_d. Inputs are natural frequency wn_d and relative damping zeta_d.
def refModel3(x_d, v_d, a_d, r, wn_d, zeta_d, v_max, sampleTime):
    
    # desired "jerk"
    j_d = wn_d**3 * (r -x_d) - (2*zeta_d+1) * wn_d**2 * v_d - (2*zeta_d+1) * wn_d * a_d

   # Forward Euler integration
    x_d += sampleTime * v_d             # desired position
    v_d += sampleTime * a_d             # desired velocity
    a_d += sampleTime * j_d             # desired acceleration 
    
    # Velocity saturation
    if (v_d > v_max):
        v_d = v_max
    elif (v_d < -v_max): 
        v_d = -v_max    
    
    return x_d, v_d, a_d

#------------------------------------------------------------------------------

def refModel(x_d, v_d, a_d, x_ref, v_max, zeta_d, w_d, h, euler_angle=True):
    """Third-order reference model (MSS refModel.m): desired position, velocity and acceleration at t + h."""
    e_x = ssa(x_d - x_ref) if euler_angle else x_d - x_ref
    a_d_dot = -w_d**3 * e_x - (2 * zeta_d + 1) * w_d**2 * v_d - (2 * zeta_d + 1) * w_d * a_d
    x_d, v_d, a_d = x_d + h * v_d, v_d + h * a_d, a_d + h * a_d_dot
    if abs(v_d) > v_max:
        v_d = np.sign(v_d) * v_max
    return x_d, v_d, a_d

#------------------------------------------------------------------------------

def LOSobserver(LOSangle, LOSrate, LOScommand, h, K_f):
    """LOS observer of MSS LOSobserver.m: angle[k+1] = angle + h (rate +
    K_f ssa(command - angle)), rate = angle - xi with the low-pass state xi
    (time constant T_f). Returns angle[k+1], rate[k+1]."""
    T_f = 1 / (K_f + 2 * math.sqrt(K_f) + 1)    # differentiator time constant
    a_k = LOSangle
    xi = a_k - LOSrate                          # differentiator state xi[k]
    LOSangle = a_k + h * (LOSrate + K_f * ssa(LOScommand - a_k))
    PHI = math.exp(-h / T_f)
    xi = PHI * xi + (1 - PHI) * a_k             # xi[k+1]
    LOSrate = LOSangle - xi
    return LOSangle, LOSrate

#------------------------------------------------------------------------------

def addIntermediateWaypoints(wpt, multiplier):
    """Each leg is cut into 'multiplier' equal parts
    (MSS addIntermediateWaypoints.m; x and y only)."""
    X, Y = wpt["x"], wpt["y"]
    xs, ys = [], []
    for i in range(len(X) - 1):
        xs.append(X[i])
        ys.append(Y[i])
        for j in range(1, multiplier):
            t = j / multiplier
            xs.append(X[i] * (1 - t) + X[i + 1] * t)
            ys.append(Y[i] * (1 - t) + Y[i + 1] * t)
    xs.append(X[-1])
    ys.append(Y[-1])
    return wpt_dict(xs, ys)

#------------------------------------------------------------------------------

class _WaypointLaw:
    """Waypoint bookkeeping shared by LOSchi, ILOSpsi and ALOSpsi (the same
    code in the three m-files): k is the active waypoint (0-based); after the
    last waypoint the path goes on along the last bearing. 'switches' holds
    the call numbers (0, 1, ...) at which k was increased."""

    def __init__(self):
        self.k = None
        self.calls = 0
        self.switches = []

    def _start(self, Delta_h, R_switch, wpt):
        X, Y = wpt["x"], wpt["y"]
        if R_switch > np.min(np.sqrt(np.diff(X)**2 + np.diff(Y)**2)):
            raise ValueError("The distances between the waypoints must be "
                             "larger than R_switch")
        if R_switch < 0:
            raise ValueError("R_switch must be larger than zero")
        if Delta_h < 0:
            raise ValueError("Delta must be larger than zero")
        self.k = 0
        self.xk, self.yk = X[0], Y[0]

    def _errors(self, x, y, R_switch, wpt):
        X, Y = wpt["x"], wpt["y"]
        n, k, xk, yk = len(X), self.k, self.xk, self.yk
        if k < n - 1:                           # next waypoint
            xn, yn = X[k + 1], Y[k + 1]
        else:                                   # go on along the last bearing
            bearing = math.atan2(Y[n - 1] - Y[n - 2], X[n - 1] - X[n - 2])
            xn = X[n - 1] + 1e10 * math.cos(bearing)
            yn = Y[n - 1] + 1e10 * math.sin(bearing)
        pi_h = math.atan2(yn - yk, xn - xk)     # path-tangential angle
        x_e = (x - xk) * math.cos(pi_h) + (y - yk) * math.sin(pi_h)
        y_e = -(x - xk) * math.sin(pi_h) + (y - yk) * math.cos(pi_h)
        d = math.sqrt((xn - xk)**2 + (yn - yk)**2)
        if d - x_e < R_switch and k < n - 1:    # along-track switching
            self.k, self.xk, self.yk = k + 1, xn, yn
            self.switches.append(self.calls)
        self.calls += 1
        return pi_h, x_e, y_e

#------------------------------------------------------------------------------

class LOSchi(_WaypointLaw):
    """chi_ref, y_e = LOSchi()(x, y, Delta_h, R_switch, wpt): proportional LOS
    for a course autopilot (MSS LOSchi.m), chi_ref = pi_h - atan(y_e/Delta_h)."""

    def __call__(self, x, y, Delta_h, R_switch, wpt):
        if self.k is None:
            self._start(Delta_h, R_switch, wpt)
        pi_h, x_e, y_e = self._errors(x, y, R_switch, wpt)
        chi_ref = pi_h - math.atan(y_e / Delta_h)
        return chi_ref, y_e

#------------------------------------------------------------------------------

class ILOSpsi(_WaypointLaw):
    """psi_ref, y_e = ILOSpsi()(x, y, Delta_h, kappa, h, R_switch, wpt):
    integral LOS of Borhaug et al. (2008) for a heading autopilot (ILOSpsi.m)."""

    def __call__(self, x, y, Delta_h, kappa, h, R_switch, wpt):
        if self.k is None:
            self._start(Delta_h, R_switch, wpt)
            self.y_int = 0.0
        pi_h, x_e, y_e = self._errors(x, y, R_switch, wpt)
        Kp, y_int = 1 / Delta_h, self.y_int
        psi_ref = pi_h - math.atan(Kp * (y_e + kappa * y_int))
        self.y_int = y_int + h * Delta_h * y_e / (
            Delta_h**2 + (y_e + kappa * y_int)**2)
        return psi_ref, y_e

#------------------------------------------------------------------------------

class ALOSpsi(_WaypointLaw):
    """psi_ref, y_e, beta_c_hat, k_active = ALOSpsi()(x, y, Delta_h, gamma_h,
    h, R_switch, wpt): adaptive LOS of Fossen (2023) for a heading autopilot
    (MSS ALOSpsi.m); k_active is 0-based."""

    def __call__(self, x, y, Delta_h, gamma_h, h, R_switch, wpt):
        if self.k is None:
            self._start(Delta_h, R_switch, wpt)
            self.beta_hat = 0.0
        pi_h, x_e, y_e = self._errors(x, y, R_switch, wpt)
        b = self.beta_hat
        psi_ref = pi_h - b - math.atan(y_e / Delta_h)
        D = Delta_h
        self.beta_hat = b + h * gamma_h * D * y_e / math.sqrt(D**2 + y_e**2)
        return psi_ref, y_e, b, self.k

#------------------------------------------------------------------------------

def proj(theta_hat, tau, M_theta):
    """Projection algorithm of MSS ALOS3D.m (eps = 0.001)."""
    M_hat = M_theta + 0.001
    if abs(theta_hat) > M_theta and theta_hat * tau > 0:
        c = min(1.0, (theta_hat**2 - M_theta**2) / (M_hat**2 - M_theta**2))
        return (1 - c) * tau
    return tau

#------------------------------------------------------------------------------

class ALOS3D:
    """psi_ref, theta_ref, y_e, z_e, alpha_c_hat, beta_c_hat = ALOS3D()(x, y,
    z, Delta_h, Delta_v, gamma_h, gamma_v, M_theta, h, R_switch, wpt): 3-D ALOS
    of Fossen and Aguiar (2024), MSS ALOS3D.m. A switch happens when the
    vehicle is inside a sphere of radius R_switch around the next waypoint;
    k is 0-based."""

    def __init__(self):
        self.k = None
        self.calls = 0
        self.switches = []

    def __call__(self, x, y, z, Delta_h, Delta_v, gamma_h, gamma_v, M_theta,
                 h, R_switch, wpt):
        X, Y, Z = wpt["x"], wpt["y"], wpt["z"]
        n = len(X)
        if self.k is None:
            dmin = np.min(np.sqrt(np.diff(X)**2 + np.diff(Y)**2
                                  + np.diff(Z)**2))
            if R_switch > dmin:
                raise ValueError("The distances between the waypoints must "
                                 "be larger than R_switch")
            self.alpha_hat, self.beta_hat = 0.0, 0.0
            self.k, self.xk, self.yk, self.zk = 0, X[0], Y[0], Z[0]
        k, xk, yk, zk = self.k, self.xk, self.yk, self.zk
        if k < n - 1:
            xn, yn, zn = X[k + 1], Y[k + 1], Z[k + 1]
        else:                                   # last bearing and depth
            bearing = math.atan2(Y[n - 1] - Y[n - 2], X[n - 1] - X[n - 2])
            xn = X[n - 1] + 1e10 * math.cos(bearing)
            yn = Y[n - 1] + 1e10 * math.sin(bearing)
            zn = Z[n - 1]
        pi_h = math.atan2(yn - yk, xn - xk)     # azimuth and elevation
        pi_v = math.atan2(-(zn - zk), math.sqrt((xn - xk)**2 + (yn - yk)**2))
        ch, sh = math.cos(pi_h), math.sin(pi_h)
        cv, sv = math.cos(pi_v), math.sin(pi_v)
        Rz = np.array([[ch, -sh, 0.0], [sh, ch, 0.0], [0.0, 0.0, 1.0]])
        Ry = np.array([[cv, 0.0, sv], [0.0, 1.0, 0.0], [-sv, 0.0, cv]])
        e = Ry.T @ Rz.T @ np.array([x - xk, y - yk, z - zk])
        y_e, z_e = e[1], e[2]
        d = math.sqrt((xn - x)**2 + (yn - y)**2 + (zn - z)**2)
        if d < R_switch and k < n - 1:          # sphere of acceptance
            self.k, self.xk, self.yk, self.zk = k + 1, xn, yn, zn
            self.switches.append(self.calls)
        self.calls += 1
        a, b = self.alpha_hat, self.beta_hat
        psi_ref = pi_h - b - math.atan(y_e / Delta_h)
        theta_ref = pi_v + a + math.atan(z_e / Delta_v)
        Dv, Dh = Delta_v, Delta_h
        self.alpha_hat = a + h * gamma_v * Dv / math.sqrt(Dv**2 + z_e**2) \
            * proj(a, z_e, M_theta)
        self.beta_hat = b + h * gamma_h * Dh / math.sqrt(Dh**2 + y_e**2) \
            * proj(b, y_e, M_theta)
        return psi_ref, theta_ref, y_e, z_e, a, b

#------------------------------------------------------------------------------

def crosstrackWpt(x2, y2, x1, y1, x, y):
    """Signed cross-track error of (x, y) from the line (x1, y1) -> (x2, y2)
    (MSS crosstrackWpt.m)."""
    pi_h = math.atan2(y2 - y1, x2 - x1)
    return -(x - x1) * math.sin(pi_h) + (y - y1) * math.cos(pi_h)

#------------------------------------------------------------------------------

def crosstrackWpt3D(p2, p1, p):
    """p_e = Ry' Rz' (p - p1) as written in MSS crosstrackWpt3D.m, including
    its R_z, whose (2,1) element is -sin(pi_h) (ALOS3D.m uses +sin(pi_h));
    check item 6 compares the two."""
    e = np.asarray(p2, float) - np.asarray(p1, float)
    pi_h = math.atan2(e[1], e[0])
    pi_v = math.atan2(-e[2], math.sqrt(e[0]**2 + e[1]**2))
    ch, sh = math.cos(pi_h), math.sin(pi_h)
    cv, sv = math.cos(pi_v), math.sin(pi_v)
    R_z = np.array([[ch, -sh, 0.0], [-sh, ch, 0.0], [0.0, 0.0, 1.0]])
    R_y = np.array([[cv, 0.0, sv], [0.0, 1.0, 0.0], [-sv, 0.0, cv]])
    return R_y.T @ R_z.T @ (np.asarray(p, float) - np.asarray(p1, float))

#------------------------------------------------------------------------------

def wpt_dict(x, y, z=None):
    """A waypoint table {'x': ..., 'y': ..., 'z': ...}
    (wpt.pos.x, wpt.pos.y, wpt.pos.z of MSS)."""
    w = {"x": np.asarray(x, float), "y": np.asarray(y, float)}
    if z is not None:
        w["z"] = np.asarray(z, float)
    return w
