#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vessel models of the MSS toolbox (MSS CRAFT folder) as functions with the MSS interface, xdot = model(x, ui, ...):
the Mariner, the container ship (nonlinear and linearized), the Zeefakkel, the DP supply vessel, the REMUS 100 AUV, ...
They return the same numbers as the m-files of the MSS version the fork is pinned to. The vehicle classes in
python_vehicle_simulator.vehicles are the PVS simulator objects and are not changed.

Each function names its MSS m-file (MSS: github.com/cybergalactic/MSS).
"""

import numpy as np
import math
from .gnc import D2R, G_MSS, Rz, crossFlowDrag, eulerang, forceLiftDrag, gRvect, quatern, sat
from .ins import gravity
from .models import Dmtrx, imlay61, spheroid
from .numerics import interp1_extrap


#------------------------------------------------------------------------------

ZEEFAKKEL = dict(L=45.0, U_range=(1.0, 7.0), n1=1.0, n3=0.4, delta_max=np.deg2rad(30), Ddelta_max=np.deg2rad(10),
                 data=np.array([[2.0, 0.15, 33.0], [2.6, 0.19, 33.0], [3.6, 0.29, 33.0], [4.0, 0.37, 33.0],
                                [5.0, 0.50, 31.0], [6.2, 0.83, 43.0]]))

#------------------------------------------------------------------------------

def zeefakkel_KT(U):
    """K (1/s) and T (s) of the Zeefakkel at the speed U (linear inter- and extrapolation, as zeefakkel.m)."""
    d = ZEEFAKKEL["data"]
    return tuple(interp1_extrap(U, d[:, 0], d[:, i]) for i in (1, 2))

#------------------------------------------------------------------------------

def zeefakkel(r, U, delta, delta_c, d_r=0.0):
    """MSS zeefakkel.m (recreational craft, L = 45 m; Van Amerongen 1982): Norrbin model
    T r' + n3 r^3 + n1 r = K delta + d_r with n1 = 1, n3 = 0.4 and speed-dependent K, T; rudder
    delta' = sat(sat(delta_c, 30 deg) - delta, 10 deg/s). Returns (psi_dot, r_dot, delta_dot)."""
    P = ZEEFAKKEL
    if U < P["U_range"][0] or U > P["U_range"][1]:
        raise ValueError("U should be between 1-7 m/s")
    K, T = zeefakkel_KT(U)
    delta_c = sat(delta_c, P["delta_max"])
    delta_dot = sat(delta_c - delta, P["Ddelta_max"])
    r_dot = (1.0 / T) * (K * delta + d_r - P["n3"] * r**3 - P["n1"] * r)
    return r, r_dot, delta_dot

#------------------------------------------------------------------------------

MARINER = dict(L=160.93, U0=7.7175, m=798e-5, Iz=39.2e-5, xG=-0.023,
               Xudot=-42e-5, Yvdot=-748e-5, Nvdot=4.646e-5, Yrdot=-9.354e-5, Nrdot=-43.8e-5,
               Xu=-184e-5, Xuu=-110e-5, Xuuu=-215e-5, Xvv=-899e-5, Xrr=18e-5, Xdd=-95e-5, Xudd=-190e-5,
               Xrv=798e-5, Xvd=93e-5, Xuvd=93e-5,
               Yv=-1160e-5, Yr=-499e-5, Yvvv=-8078e-5, Yvvr=15356e-5, Yvu=-1160e-5, Yru=-499e-5, Yd=278e-5,
               Yddd=-90e-5, Yud=556e-5, Yuud=278e-5, Yvdd=-4e-5, Yvvd=1190e-5, Y0=-4e-5, Y0u=-8e-5, Y0uu=-4e-5,
               Nv=-264e-5, Nr=-166e-5, Nvvv=1636e-5, Nvvr=-5483e-5, Nvu=-264e-5, Nru=-166e-5, Nd=-139e-5,
               Nddd=45e-5, Nud=-278e-5, Nuud=-139e-5, Nvdd=13e-5, Nvvd=-489e-5, N0=3e-5, N0u=6e-5, N0uu=3e-5)

#------------------------------------------------------------------------------

def mariner(x, ui, U0=7.7175):
    """The Mariner-class cargo ship, L = 160.93 m (MSS mariner.m). x = [u v r x y psi delta] with u, v the
    perturbations about U0 = 15 kn; ui = rudder command, positive to starboard (mariner.m sets
    its state delta_c = -ui). Rudder limits 40 deg and 5 deg/s. Returns (xdot, U)."""
    p = MARINER
    L = p["L"]
    ui = float(np.atleast_1d(ui)[0])
    U = np.sqrt((U0 + x[0])**2 + x[1]**2)
    delta_c = -ui
    u, v, r, psi, delta = x[0] / U, x[1] / U, x[2] * L / U, x[5], x[6]
    m, Iz, xG = p["m"], p["Iz"], p["xG"]
    m11, m22 = m - p["Xudot"], m - p["Yvdot"]
    m23, m32, m33 = m * xG - p["Yrdot"], m * xG - p["Nvdot"], Iz - p["Nrdot"]
    dmax, ddmax = 40.0 * np.pi / 180, 5.0 * np.pi / 180
    if abs(delta_c) >= dmax:
        delta_c = np.sign(delta_c) * dmax
    delta_dot = delta_c - delta
    if abs(delta_dot) >= ddmax:
        delta_dot = np.sign(delta_dot) * ddmax
    X = (p["Xu"] * u + p["Xuu"] * u**2 + p["Xuuu"] * u**3 + p["Xvv"] * v**2 + p["Xrr"] * r**2 + p["Xrv"] * r * v
         + p["Xdd"] * delta**2 + p["Xudd"] * u * delta**2 + p["Xvd"] * v * delta + p["Xuvd"] * u * v * delta)
    Y = (p["Yv"] * v + p["Yr"] * r + p["Yvvv"] * v**3 + p["Yvvr"] * v**2 * r + p["Yvu"] * v * u + p["Yru"] * r * u
         + p["Yd"] * delta + p["Yddd"] * delta**3 + p["Yud"] * u * delta + p["Yuud"] * u**2 * delta
         + p["Yvdd"] * v * delta**2 + p["Yvvd"] * v**2 * delta + (p["Y0"] + p["Y0u"] * u + p["Y0uu"] * u**2))
    N = (p["Nv"] * v + p["Nr"] * r + p["Nvvv"] * v**3 + p["Nvvr"] * v**2 * r + p["Nvu"] * v * u + p["Nru"] * r * u
         + p["Nd"] * delta + p["Nddd"] * delta**3 + p["Nud"] * u * delta + p["Nuud"] * u**2 * delta
         + p["Nvdd"] * v * delta**2 + p["Nvvd"] * v**2 * delta + (p["N0"] + p["N0u"] * u + p["N0uu"] * u**2))
    det = m22 * m33 - m23 * m32
    xdot = np.array([X * (U**2 / L) / m11,
                     -(-m33 * Y + m23 * N) * (U**2 / L) / det,
                     (-m32 * Y + m22 * N) * (U**2 / L**2) / det,
                     (np.cos(psi) * (U0 / U + u) - np.sin(psi) * v) * U,
                     (np.sin(psi) * (U0 / U + u) + np.cos(psi) * v) * U,
                     r * (U / L),
                     delta_dot])
    return xdot, U

#------------------------------------------------------------------------------

CONT = dict(
    L=175.0, n_max=160.0,
    m=0.00792, mx=0.000238, my=0.007049, Ix=0.0000176, alphay=0.05, lx=0.0313, ly=0.0313, Iz=0.000456,
    Jx=0.0000034, Jz=0.000419, xG=0.0,
    B=25.40, dF=8.00, g=9.81, dA=9.00, d=8.50, nabla=21222.0, KM=10.39, KB=4.6154, AR=33.0376,
    Delta=1.8219, D=6.533, GM=0.3 / 175.0, rho=1025.0, t=0.175,
    Xuu=-0.0004226, Xvr=-0.00311, Xrr=0.00020, Xphiphi=-0.00020, Xvv=-0.00386,
    Kv=0.0003026, Kr=-0.000063, Kp=-0.0000075, Kphi=-0.000021, Kvvv=0.002843, Krrr=-0.0000462,
    Kvvr=-0.000588, Kvrr=0.0010565, Kvvphi=-0.0012012, Kvphiphi=-0.0000793, Krrphi=-0.000243, Krphiphi=0.00003569,
    Yv=-0.0116, Yr=0.00242, Yp=0.0, Yphi=-0.000063, Yvvv=-0.109, Yrrr=0.00177,
    Yvvr=0.0214, Yvrr=-0.0405, Yvvphi=0.04605, Yvphiphi=0.00304, Yrrphi=0.009325, Yrphiphi=-0.001368,
    Nv=-0.0038545, Nr=-0.00222, Np=0.000213, Nphi=-0.0001424, Nvvv=0.001492, Nrrr=-0.00229,
    Nvvr=-0.0424, Nvrr=0.00156, Nvvphi=-0.019058, Nvphiphi=-0.0053766, Nrrphi=-0.0038592, Nrphiphi=0.0024195,
    kk=0.631, epsilon=0.921, xR=-0.5, wp=0.184, tau=1.09, xp=-0.526, cpv=0.0, cpr=0.0, ga=0.088,
    cRr=-0.156, cRrrr=-0.275, cRrrv=1.96, cRX=0.71, aH=0.237, zR=0.033, xH=-0.48)

#------------------------------------------------------------------------------

def _container_core(x, ui, delta_max, Ddelta_max, uP_cos, K_wave):
    """Shared core of container.m and RRDcontainer (exRRD2.m): x = [u v r x y psi p phi delta n (xw1 xw2)],
    ui = [delta_c, n_c (rpm)]. uP_cos: RRDcontainer writes uP = cos(v) (...) where container.m writes u (...);
    K_wave is the wave roll moment added to K (nondimensional). Returns the first ten state derivatives and U."""
    P = CONT
    L = P["L"]
    U = np.sqrt(x[0]**2 + x[1]**2)
    if U <= 0:
        raise ValueError("The ship must have speed greater than zero")
    if x[9] <= 0:
        raise ValueError("The propeller rpm must be greater than zero")
    delta_c = ui[0]
    n_c = ui[1] / 60 * L / U
    u, v = x[0] / U, x[1] / U
    p, r = x[6] * L / U, x[2] * L / U
    phi, psi = x[7], x[5]
    delta, n = x[8], x[9] / 60 * L / U
    m, mx, my = P["m"], P["mx"], P["my"]
    W = P["rho"] * P["g"] * P["nabla"] / (P["rho"] * L**2 * U**2 / 2)
    m11, m22, m32, m42 = m + mx, m + my, -my * P["ly"], my * P["alphay"]
    m33, m44 = P["Ix"] + P["Jx"], P["Iz"] + P["Jz"]
    if abs(delta_c) >= delta_max * np.pi / 180:
        delta_c = np.sign(delta_c) * delta_max * np.pi / 180
    delta_dot = delta_c - delta
    if abs(delta_dot) >= Ddelta_max * np.pi / 180:
        delta_dot = np.sign(delta_dot) * Ddelta_max * np.pi / 180
    n_c, n = n_c * U / L, n * U / L
    if abs(n_c) >= P["n_max"] / 60:
        n_c = np.sign(n_c) * P["n_max"] / 60
    Tm = 5.65 / n if n > 0.3 else 18.83
    n_dot = 1 / Tm * (n_c - n) * 60
    vR = P["ga"] * v + P["cRr"] * r + P["cRrrr"] * r**3 + P["cRrrv"] * r**2 * v
    uP = (np.cos(v) if uP_cos else u) * ((1 - P["wp"]) + P["tau"] * ((v + P["xp"] * r)**2
                                                                     + P["cpv"] * v + P["cpr"] * r))
    J = uP * U / (n * P["D"])
    KT = 0.527 - 0.455 * J
    uR = uP * P["epsilon"] * np.sqrt(1 + 8 * P["kk"] * KT / (np.pi * J**2))
    alphaR = delta + np.arctan(vR / uR)
    FN = -((6.13 * P["Delta"]) / (P["Delta"] + 2.25)) * (P["AR"] / L**2) * (uR**2 + vR**2) * np.sin(alphaR)
    T = 2 * P["rho"] * P["D"]**4 / (U**2 * L**2 * P["rho"]) * KT * n * abs(n)
    X = (P["Xuu"] * u**2 + (1 - P["t"]) * T + P["Xvr"] * v * r + P["Xvv"] * v**2 + P["Xrr"] * r**2
         + P["Xphiphi"] * phi**2 + P["cRX"] * FN * np.sin(delta) + (m + my) * v * r)
    Y = (P["Yv"] * v + P["Yr"] * r + P["Yp"] * p + P["Yphi"] * phi + P["Yvvv"] * v**3 + P["Yrrr"] * r**3
         + P["Yvvr"] * v**2 * r + P["Yvrr"] * v * r**2 + P["Yvvphi"] * v**2 * phi + P["Yvphiphi"] * v * phi**2
         + P["Yrrphi"] * r**2 * phi + P["Yrphiphi"] * r * phi**2 + (1 + P["aH"]) * FN * np.cos(delta)
         - (m + mx) * u * r)
    K = (P["Kv"] * v + P["Kr"] * r + P["Kp"] * p + P["Kphi"] * phi + P["Kvvv"] * v**3 + P["Krrr"] * r**3
         + P["Kvvr"] * v**2 * r + P["Kvrr"] * v * r**2 + P["Kvvphi"] * v**2 * phi + P["Kvphiphi"] * v * phi**2
         + P["Krrphi"] * r**2 * phi + P["Krphiphi"] * r * phi**2 - (1 + P["aH"]) * P["zR"] * FN * np.cos(delta)
         + mx * P["lx"] * u * r - W * P["GM"] * phi)
    N = (P["Nv"] * v + P["Nr"] * r + P["Np"] * p + P["Nphi"] * phi + P["Nvvv"] * v**3 + P["Nrrr"] * r**3
         + P["Nvvr"] * v**2 * r + P["Nvrr"] * v * r**2 + P["Nvvphi"] * v**2 * phi + P["Nvphiphi"] * v * phi**2
         + P["Nrrphi"] * r**2 * phi + P["Nrphiphi"] * r * phi**2 + (P["xR"] + P["aH"] * P["xH"]) * FN * np.cos(delta))
    K = K + K_wave
    detM = m22 * m33 * m44 - m32**2 * m44 - m42**2 * m33
    xdot = np.array([
        X * (U**2 / L) / m11,
        -((-m33 * m44 * Y + m32 * m44 * K + m42 * m33 * N) / detM) * (U**2 / L),
        ((-m42 * m33 * Y + m32 * m42 * K + N * m22 * m33 - N * m32**2) / detM) * (U**2 / L**2),
        (np.cos(psi) * u - np.sin(psi) * np.cos(phi) * v) * U,
        (np.sin(psi) * u + np.cos(psi) * np.cos(phi) * v) * U,
        np.cos(phi) * r * (U / L),
        ((-m32 * m44 * Y + K * m22 * m44 - K * m42**2 + m32 * m42 * N) / detM) * (U**2 / L**2),
        p * (U / L),
        delta_dot,
        n_dot])
    return xdot, U

#------------------------------------------------------------------------------

def container(x, ui):
    """MSS container.m (Son and Nomoto 1982), x = [u v r x y psi p phi delta n] (m/s, rad/s, m, rad, rpm),
    ui = [delta_c (rad), n_c (rpm)]; rudder 10 deg and 5 deg/s. Returns (xdot, U)."""
    x, ui = np.asarray(x, float), np.asarray(ui, float)
    if len(x) != 10:
        raise ValueError("x-vector must have dimension 10 !")
    return _container_core(x, ui, 10.0, 5.0, False, 0.0)

#------------------------------------------------------------------------------

def RRDcontainer(x, ui, w_noise):
    """RRDcontainer, the local function of exRRD2.m: container.m with rudder 20 deg and 20 deg/s, uP written with
    cos(v), and a second-order wave model xw' = Aw xw + Bw w (Kw = 0.0001, w_wave = 1 rad/s, zeta 0.1) whose first
    state is added to the roll moment K; w_noise is the randn(1) of the m-file. x has 12 states. Returns (xdot, U)."""
    x = np.asarray(x, float)
    if len(x) != 12:
        raise ValueError("x-vector must have dimension 12 !")
    Kw, w_wave = 0.0001, 1.0
    Aw = np.array([[0.0, 1.0], [-w_wave**2, -2 * 0.1 * w_wave]])
    Bw = np.array([0.0, Kw])
    xdot10, U = _container_core(x, np.asarray(ui, float), 20.0, 20.0, True, x[10])
    return np.r_[xdot10, Aw @ x[10:12] + Bw * w_noise], U

#------------------------------------------------------------------------------

LCONT = dict(M=np.array([[0.01497, 0.0003525, -0.0002205],
                         [0.0003525, 0.000875, 0.0],
                         [-0.0002205, 0.0, 0.0000210]]),
             N=np.array([[0.012035, 0.00522, 0.0],
                         [0.0038436, 0.00243, -0.000213],
                         [-0.000314, 0.0000692, 0.0000075]]),
             G=np.array([[0.0, 0.0, 0.0000704],
                         [0.0, 0.0, 0.0001468],
                         [0.0, 0.0, 0.0004966]]),
             b=np.array([-0.002578, 0.00126, 0.0000855]))

#------------------------------------------------------------------------------

def Lcontainer(x, ui, U0=7.0):
    """Linearized container ship of MSS Lcontainer.m: x = [u v r x y psi p phi delta], nu = [v r p],
    eta = [y psi phi], ui = delta_c (rad); rudder 10 deg and 5 deg/s. Returns (xdot, U)."""
    x = np.asarray(x, float)
    if len(x) != 9:
        raise ValueError("x-vector must have dimension 9!")
    L = 175.0
    U = np.sqrt(U0**2 + x[1]**2)
    delta_c = float(np.atleast_1d(ui)[0])
    v, y, r, psi, p, phi, delta = x[1], x[4], x[2], x[5], x[6], x[7], x[8]
    nu, eta = np.array([v, r, p]), np.array([y, psi, phi])
    T, Tinv = np.diag([1.0, 1 / L, 1 / L]), np.diag([1.0, L, L])
    if abs(delta_c) >= 10 * D2R:
        delta_c = np.sign(delta_c) * 10 * D2R
    delta_dot = delta_c - delta
    if abs(delta_dot) >= 5 * D2R:
        delta_dot = np.sign(delta_dot) * 5 * D2R
    M, N, G, b = LCONT["M"], LCONT["N"], LCONT["G"], LCONT["b"]
    nudot = np.linalg.inv(T @ M @ Tinv) @ ((U**2 / L) * T @ b * delta - (U / L) * T @ N @ Tinv @ nu
                                           - (U / L)**2 * T @ G @ Tinv @ eta)
    xdot = np.array([0.0, nudot[0], nudot[1], np.cos(psi) * U0 - np.sin(psi) * np.cos(phi) * v,
                     np.sin(psi) * U0 + np.cos(psi) * np.cos(phi) * v, np.cos(phi) * r, nudot[2], p, delta_dot])
    return xdot, U

#------------------------------------------------------------------------------

SUPPLY_DP = dict(L=76.2, g=9.81, mass=6000e3,
                 Mbis=np.array([[1.1274, 0.0, 0.0], [0.0, 1.8902, -0.0744], [0.0, -0.0744, 0.1278]]),
                 Dbis=np.array([[0.0358, 0.0, 0.0], [0.0, 0.1183, -0.0124], [0.0, -0.0041, 0.0308]]))

#------------------------------------------------------------------------------

def supply_matrices():
    """M and D of supply.m: M = m T^-2 (T Mbis T^-1), D = m T^-2 (sqrt(g/L) T Dbis T^-1), T = diag(1, 1, 1/L)."""
    P = SUPPLY_DP
    L, g, m = P["L"], P["g"], P["mass"]
    T, Tinv = np.diag([1.0, 1.0, 1.0 / L]), np.diag([1.0, 1.0, L])
    M = m * Tinv @ Tinv @ (T @ P["Mbis"] @ Tinv)
    D = m * Tinv @ Tinv @ (np.sqrt(g / L) * T @ P["Dbis"] @ Tinv)
    return M, D

#------------------------------------------------------------------------------

def supply(x, tau):
    """Linear DP model of a supply vessel, L = 76.2 m (MSS supply.m): x = [x, y, psi, u, v, r],
    xdot = [R(psi) nu; M^-1 (tau - D nu)]. Returns (xdot, U)."""
    x = np.asarray(x, float)
    M, D = supply_matrices()
    nu = x[3:6]
    xdot = np.r_[Rz(x[2]) @ nu, np.linalg.solve(M, np.asarray(tau, float) - D @ nu)]
    return xdot, np.hypot(x[3], x[4])

#------------------------------------------------------------------------------

REMUS = dict(L=1.6, D=0.19, scale=1.0096, r44=0.3, r_bG=np.array([0.0, 0.0, 0.02]), r_bB=np.zeros(3),
             Cd=0.42, D_prop=0.14, t_prop=0.1, Ja_max=0.6632, KT_0=0.4566, KQ_0=0.0700,
             KT_max=0.1798, KQ_max=0.0312, S_fin=0.00665, CL_delta_r=0.5, CL_delta_s=0.7,
             T1=20.0, T2=20.0, zeta4=0.3, zeta5=0.8, T6=1.0, delta_max=20 * D2R, n_max=1525.0,
             rho=1026.0, mu=63.446827 * D2R)

#------------------------------------------------------------------------------

# C_A terms that remus100.m sets to zero: (5,3), (3,5), (5,1), (1,5), (6,1), (1,6), (6,2), (2,6) (MATLAB indices)
MUNK_TERMS = ((4, 2), (2, 4), (4, 0), (0, 4), (5, 0), (0, 5), (5, 1), (1, 5))

#------------------------------------------------------------------------------

def remus100_parts(x, ui, Vc=0.0, betaVc=0.0, w_c=0.0, keep_munk=False, re_length=None):
    """REMUS 100 model of MSS remus100.m with every term returned in a dict: M, C, D, tau (propeller and fins),
    tau_liftdrag, tau_crossflow, g, nu_r, nu_c, Dnu_c, J, alpha, U_r, U and the propeller values."""
    P = REMUS
    x = np.asarray(x, float)
    ui = np.asarray(ui, float)
    if len(ui) != 3:
        raise ValueError("u-vector must have dimension 3!")
    if len(x) not in (12, 13):
        raise ValueError("x-vector must have dimension 12 or 13")
    g_mu = gravity(P["mu"])
    rho = P["rho"]
    nu = x[0:6]
    psi = x[11] if len(x) == 12 else np.arctan2(2 * (x[9] * x[12] + x[10] * x[11]), 1 - 2 * (x[11]**2 + x[12]**2))
    delta_r = sat(ui[0], P["delta_max"])
    delta_s = sat(ui[1], P["delta_max"])
    n_p = sat(ui[2], P["n_max"]) / 60.0
    u_c, v_c = Vc * np.cos(betaVc - psi), Vc * np.sin(betaVc - psi)
    nu_c = np.array([u_c, v_c, w_c, 0.0, 0.0, 0.0])
    Dnu_c = np.array([nu[5] * v_c, -nu[5] * u_c, 0.0, 0.0, 0.0, 0.0])
    nu_r = nu - nu_c
    alpha = np.arctan2(nu_r[2], nu_r[0])
    U_r = np.sqrt(nu_r[0]**2 + nu_r[1]**2 + nu_r[2]**2)
    U = np.sqrt(nu[0]**2 + nu[1]**2 + nu[2]**2)
    L_auv, D_auv = P["L"], P["D"]
    S = 0.7 * L_auv * D_auv
    a, b = P["scale"] * L_auv / 2, P["scale"] * D_auv / 2
    r_bG, r_bB = P["r_bG"], P["r_bB"]
    CD_0 = P["Cd"] * np.pi * b**2 / S
    D_prop, Ja_max = P["D_prop"], P["Ja_max"]
    Va = 0.944 * U_r
    KT_0, KQ_0, KT_max, KQ_max = P["KT_0"], P["KQ_0"], P["KT_max"], P["KQ_max"]
    if n_p > 0:
        X_prop = rho * D_prop**4 * (KT_0 * abs(n_p) * n_p + (KT_max - KT_0) / Ja_max * (Va / D_prop) * abs(n_p))
        K_prop = rho * D_prop**5 * (KQ_0 * abs(n_p) * n_p + (KQ_max - KQ_0) / Ja_max * (Va / D_prop) * abs(n_p))
    else:
        X_prop = rho * D_prop**4 * KT_0 * abs(n_p) * n_p
        K_prop = rho * D_prop**5 * KQ_0 * abs(n_p) * n_p
    A_r = A_s = 2 * P["S_fin"]
    x_r = x_s = -a
    MRB, CRB = spheroid(a, b, nu[3:6], r_bG)          # rigid body about the CO
    MA, CA = imlay61(a, b, nu_r, P["r44"])            # added mass, Lamb's k-factors
    if not keep_munk:
        for i, j in MUNK_TERMS:                       # set to zero in remus100.m
            CA[i, j] = 0.0
    M = MRB + MA
    C = CRB + CA
    m = MRB[0, 0]
    W = m * g_mu
    B = W                                             # neutrally buoyant
    hs = np.concatenate(([W], r_bG, r_bB))
    D = Dmtrx([P["T1"], P["T2"], P["T6"]], [P["zeta4"], P["zeta5"]], MRB, MA, hs)
    D[0, 0] *= np.exp(-3 * U_r)                       # vanishes at high speed
    tau_liftdrag = forceLiftDrag(D_auv, S, CD_0, alpha, U_r)
    tau_crossflow = crossFlowDrag(L_auv, D_auv, D_auv, nu_r, "cylinder",
                                  re_length=re_length)
    if len(x) == 13:
        J, R = quatern(x[9:13])[0:2]                  # unit quaternion
    else:
        J, R = eulerang(x[9], x[10], x[11])[0:2]      # Euler angles
    g = gRvect(W, B, R, r_bG, r_bB)
    U_rh = np.sqrt(nu_r[0]**2 + nu_r[1]**2)
    U_rv = np.sqrt(nu_r[0]**2 + nu_r[2]**2)
    X_r = -0.5 * rho * U_rh**2 * A_r * P["CL_delta_r"] * delta_r**2
    X_s = -0.5 * rho * U_rv**2 * A_s * P["CL_delta_s"] * delta_s**2
    Y_r = -0.5 * rho * U_rh**2 * A_r * P["CL_delta_r"] * delta_r
    Z_s = -0.5 * rho * U_rv**2 * A_s * P["CL_delta_s"] * delta_s
    tau = np.array([(1 - P["t_prop"]) * X_prop + X_r + X_s, Y_r, Z_s, K_prop / 10, -x_s * Z_s, x_r * Y_r])
    return dict(M=M, MRB=MRB, MA=MA, C=C, CRB=CRB, CA=CA, D=D, tau=tau, tau_liftdrag=tau_liftdrag,
                tau_crossflow=tau_crossflow, g=g, nu=nu, nu_r=nu_r, nu_c=nu_c, Dnu_c=Dnu_c, J=J, R=R, alpha=alpha,
                U_r=U_r, U=U, W=W, B=B, m=m, X_prop=X_prop, K_prop=K_prop, a=a, b=b, S=S, CD_0=CD_0,
                X_r=X_r, X_s=X_s, Y_r=Y_r, Z_s=Z_s)

#------------------------------------------------------------------------------

def remus100(x=None, ui=None, Vc=0.0, betaVc=0.0, w_c=0.0, keep_munk=False, re_length=None):
    """[xdot, U, M] = remus100(x, ui, Vc, betaVc, w_c) of MSS: x = [u v w p q r x y z phi theta psi] (12 states) or
    [u v w p q r x y z eta eps1 eps2 eps3] (13 states), ui = [delta_r, delta_s, n] (rad, rad, rpm). Ocean current:
    speed Vc (m/s) towards direction betaVc (rad) in NED, vertical current w_c (m/s). remus100() returns M."""
    if x is None:
        x, ui = np.zeros(12), np.zeros(3)
    p = remus100_parts(x, ui, Vc, betaVc, w_c, keep_munk, re_length)
    rhs = (p["tau"] + p["tau_liftdrag"] + p["tau_crossflow"]
           - p["C"] @ p["nu_r"] - p["D"] @ p["nu_r"] - p["g"])
    nu_dot = p["Dnu_c"] + np.linalg.solve(p["M"], rhs)
    xdot = np.concatenate((nu_dot, p["J"] @ p["nu"]))
    return xdot, p["U"], p["M"]
