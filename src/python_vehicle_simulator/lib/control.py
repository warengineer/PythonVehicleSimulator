#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Control methods.

Reference: T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and
Motion Control. 2nd. Edition, Wiley. 
URL: www.fossen.biz/wiley

Author:     Thor I. Fossen
"""

import numpy as np
from python_vehicle_simulator.lib.guidance import refModel3
from python_vehicle_simulator.lib.gnc import ssa, Rzyx
from .gnc import Rz
from .numerics import lqr

# SISO PID pole placement
def PIDpolePlacement(
    e_int,
    e_x,
    e_v,
    x_d,
    v_d,
    a_d,
    m,
    d,
    k,
    wn_d,
    zeta_d,
    wn,
    zeta,
    r,
    v_max,
    sampleTime,
):

    # PID gains based on pole placement
    Kp = m * wn ** 2.0 - k
    Kd = m * 2.0 * zeta * wn - d
    Ki = (wn / 10.0) * Kp

    # PID control law
    u = -Kp * e_x - Kd * e_v - Ki * e_int

    # Integral error, Euler's method
    e_int += sampleTime * e_x

    # 3rd-order reference model for smooth position, velocity and acceleration
    [x_d, v_d, a_d] = refModel3(x_d, v_d, a_d, r, wn_d, zeta_d, v_max, sampleTime)

    return u, e_int, x_d, v_d, a_d


# MIMO nonlinear PID pole placement
def DPpolePlacement(
    e_int, M3, D3, eta3, nu3, x_d, y_d, psi_d, wn, zeta, eta_ref, sampleTime
):

    # PID gains based on pole placement
    M3_diag = np.diag(np.diag(M3))
    D3_diag = np.diag(np.diag(D3))
    
    Kp = wn @ wn @ M3_diag
    Kd = 2.0 * zeta @ wn @ M3_diag - D3_diag
    Ki = (1.0 / 10.0) * wn @ Kp

    # DP control law - setpoint regulation
    e = eta3 - np.array([x_d, y_d, psi_d])
    e[2] = ssa(e[2])
    R = Rzyx(0.0, 0.0, eta3[2])
    tau = (
        - np.matmul((R.T @ Kp), e)
        - np.matmul(Kd, nu3)
        - np.matmul((R.T @ Ki), e_int)
    )

    # Low-pass filters, Euler's method
    T = 5.0 * np.array([1 / wn[0][0], 1 / wn[1][1], 1 / wn[2][2]])
    x_d += sampleTime * (eta_ref[0] - x_d) / T[0]
    y_d += sampleTime * (eta_ref[1] - y_d) / T[1]
    psi_d += sampleTime * (eta_ref[2] - psi_d) / T[2]

    # Integral error, Euler's method
    e_int += sampleTime * e

    return tau, e_int, x_d, y_d, psi_d

# Heading autopilot - Intergral SMC (Equation 16.479 in Fossen 2021)
def integralSMC(
    e_int,
    e_x,
    e_v,
    x_d,
    v_d,
    a_d,
    T_nomoto,
    K_nomoto,
    wn_d,
    zeta_d,
    K_d,
    K_sigma,
    lam,
    phi_b,
    r,
    v_max,
    sampleTime,
):

    # Sliding surface
    v_r_dot = a_d - 2 * lam * e_v - lam ** 2 * ssa(e_x)
    v_r     = v_d - 2 * lam * ssa(e_x) - lam ** 2 * e_int
    sigma   = e_v + 2 * lam * ssa(e_x) + lam ** 2 * e_int

    #  Control law
    if abs(sigma / phi_b) > 1.0:
        delta = ( T_nomoto * v_r_dot + v_r - K_d * sigma 
                 - K_sigma * np.sign(sigma) ) / K_nomoto
    else:
        delta = ( T_nomoto * v_r_dot + v_r - K_d * sigma 
                 - K_sigma * (sigma / phi_b) ) / K_nomoto

    # Integral error, Euler's method
    e_int += sampleTime * ssa(e_x)

    # 3rd-order reference model for smooth position, velocity and acceleration
    [x_d, v_d, a_d] = refModel3(x_d, v_d, a_d, r, wn_d, zeta_d, v_max, sampleTime)

    return delta, e_int, x_d, v_d, a_d

#------------------------------------------------------------------------------

def lqtracker(A, B, C, Q, R):
    """LQ tracker of MSS lqtracker.m: u = G1 x + G2 y_d with G1 = -R^-1 B' P and
    G2 = -R^-1 B' ((A + B G1)')^-1 C' Q (P from lqr(A, B, C' Q C, R))."""
    A = np.atleast_2d(np.asarray(A, float))
    B = np.asarray(B, float).reshape(A.shape[0], -1)
    C = np.atleast_2d(np.asarray(C, float))
    Q, R = np.atleast_2d(np.asarray(Q, float)), np.atleast_2d(np.asarray(R, float))
    _, P, _ = lqr(A, B, C.T @ Q @ C, R)
    G1 = -np.linalg.inv(R) @ B.T @ P
    G2 = -np.linalg.inv(R) @ B.T @ np.linalg.inv((A + B @ G1).T) @ C.T @ Q
    return G1, G2

#------------------------------------------------------------------------------

class IntegralSMCheading:
    """MSS integralSMCheading.m (the persistent integral state as an attribute): sigma = r - r_r with
    r_r = r_d - 2 lam ssa(psi - psi_d) - lam^2 psi_int, delta = (T r_r' + r_r - K_d sigma - K_sigma sat(sigma/phi_b))/K
    (sat as sgn outside the boundary layer); psi_int is propagated by Euler after delta is computed."""

    def __init__(self):
        self.psi_int = 0.0

    def __call__(self, psi, r, psi_d, r_d, a_d, K_d, K_sigma, lam, phi_b, K_nomoto, T_nomoto, h):
        e_psi = ssa(psi - psi_d)
        e_r = r - r_d
        r_r_dot = a_d - 2 * lam * e_r - lam**2 * e_psi
        r_r = r_d - 2 * lam * e_psi - lam**2 * self.psi_int
        sigma = r - r_r
        if abs(sigma / phi_b) > 1:
            delta = (T_nomoto * r_r_dot + r_r - K_d * sigma - K_sigma * np.sign(sigma)) / K_nomoto
        else:
            delta = (T_nomoto * r_r_dot + r_r - K_d * sigma - K_sigma * (sigma / phi_b)) / K_nomoto
        self.psi_int = self.psi_int + h * e_psi
        return delta

#------------------------------------------------------------------------------

class PIDnonlinearMIMO:
    """Nonlinear MIMO PID for DP (MSS PIDnonlinearMIMO.m, Algorithm 15.2 of Fossen 2027):
    tau = -R(psi)' (Kp e + Ki z_int) - Kd nu, e = eta - eta_d (ssa in yaw), Kp = M_diag wn^2, Kd = 2 M_diag zeta wn
    (no -D term), Ki = Kp wn/10; z_int and the setpoint filter eta_d (time constant T_f) are updated by Euler after
    tau. The persistent variables of the m-file are attributes: create a new object instead of
    'clear PIDnonlinearMIMO'. 3-DOF inputs (eta, nu, M of surge, sway, yaw) or 6-DOF inputs as in MSS.
    ki_scale (not in MSS) scales Ki; 0 gives the PD controller."""

    def __init__(self, ki_scale=1.0):
        self.eta_d = np.zeros(3)
        self.z_int = np.zeros(3)
        self.ki_scale = ki_scale          # not in MSS: 0 gives the PD controller

    def __call__(self, eta, nu, eta_ref, M, wn, zeta, T_f, h):
        eta, nu, M = np.asarray(eta, float), np.asarray(nu, float), np.asarray(M, float)
        six = len(nu) == 6
        if six:
            eta = eta[[0, 1, 5]]
            nu = nu[[0, 1, 5]]
            M = M[np.ix_([0, 1, 5], [0, 1, 5])]
        R = np.array([[np.cos(eta[2]), -np.sin(eta[2]), 0.0],
                      [np.sin(eta[2]), np.cos(eta[2]), 0.0],
                      [0.0, 0.0, 1.0]])
        M_diag = np.diag(np.diag(M))
        Kp = M_diag * wn * wn
        Kd = M_diag * 2 * zeta * wn
        Ki = 1 / 10 * Kp * wn * self.ki_scale
        e = eta - self.eta_d
        e[2] = ssa(e[2])
        tau_PID = -R.T @ (Kp @ e + Ki @ self.z_int) - Kd @ nu
        self.z_int = self.z_int + h * (eta - self.eta_d)
        eta_ref = np.asarray(eta_ref, float).ravel()
        self.eta_d = self.eta_d + h * (eta_ref - self.eta_d) / T_f
        if six:
            return np.array([tau_PID[0], tau_PID[1], 0, 0, 0, tau_PID[2]])
        return tau_PID
