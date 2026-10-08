#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vehicle models.           

Author:     Thor I. Fossen
"""

import numpy as np
import math
from .gnc import Hmtrx, Smtrx, m2c

def clarke83(U,L,B,T,Cb,R66,xg,T_surge):
    """ 
    [M,N] = clarke83(U,L,B,T,Cb,R66,xg,T_surge) computes the system matrices 
    of a linear maneuvering model based on Clarke et al. (1983). The  
    hydrodynamic derivatives are based on multiple  linear regression from two 
    sets of model tests. The first data set (Yv, Yr, Nv, Nr) is obtained from 
    rotating arm model experiments, while the second data set 
    (Yvdot, Yrdot, Nvdot, Nrdot, Yv, Yr, Nv, Nr) was obtained from a PMM  model.
    
    Outputs: 3x3 model matrices M and N in surge, sway and yaw

       M nu_dot + N(U) nu = tau,     where N(U) = C(U) + D
 
    corresponding to the linear maneuvering model
 
    (m - Xudot) u_dot - Xu u                            = (1-t) T
    (m - Yvdot) v_dot + (m - Yrdot)  r_dot - Yv v - Yr r = Yd delta
    (m - Yvdot) v_dot + (Iz - Nrdot) r_dot - Nv v - Nr r = Nd delta

    Note that the coefficients Yv, Yr, Nv and Nr in the N(U) matrix includes 
    linear damping D and the linearized Coriolis and centripetal matrix C(U).
    
    Inputs:

    U:   speed (m/s)
    L:   length (m)
    B:   beam (m)
    T:   draft (m)
    Cb:  block coefficient (-), Cb = V / (L*B*T) where V is the displaced volume
    R66: radius of gyration in yaw (smaller vessels R66 ≈ 0.25L, tankers R66 ≈ 0.27L)
    xg:  x-coordinate of the CG (m)
    T_surge: time constant in surge (s)

    Reference:  CLARKE, D., GEDLING, P. and HINE. G. (1983). The application of 
    manoeuvring criteria in hull design using linear thory. Trans.  R. lnsm nav. 
    Archit.  125, 45-68. 
    """

    # Rigid-body parameters
    rho = 1025                      # density of water
    V = Cb * L * B * T              # volume displacment
    m = rho * V                     # mass
    Iz = m * R66**2 + m * xg**2     # moment of inerta about the CO
    
    # Rigid-body inertia matrix
    MRB = np.array([
        [ m, 0 ,0], 
        [ 0, m ,m*xg], 
        [ 0, m*xg ,Iz] ],float)

    # Nondimenisonal hydrodynamic derivatives in surge
    Xudot = -0.1 * m
    U += 0.001                 # avoid the singularity for U = 0
    Xu = -( ( m-Xudot )/T_surge ) / ( 0.5 * rho * L**2 * U )
    Xudot = Xudot / ( 0.5 * rho * L**3 )

    # Nondimenisonal hydrodynamic derivatives in sway and yaw
    # from Clarke et al. (1983)
    S = math.pi * (T/L)**2                 # scale factor

    Yvdot = -S * ( 1 + 0.16 * Cb * B/T - 5.1 * (B/L)**2 )
    Yrdot = -S * ( 0.67 * B/L - 0.0033 * (B/T)**2 )
    Nvdot = -S * ( 1.1 * B/L - 0.041 * (B/T) )
    Nrdot = -S * ( 1/12 + 0.017 * Cb * (B/T) - 0.33 * (B/L) )
    Yv = -S * ( 1 + 0.4 * Cb * (B/T) )
    Yr = -S * ( -1/2 + 2.2 * (B/L) - 0.08 * (B/T) )
    Nv = -S * ( 1/2 + 2.4 * (T/L) )
    Nr = -S * ( 1/4 + 0.039 * (B/T) - 0.56 * (B/L) )

    # Nondimenisonal hydrodynamic matrices 
    MA_prime = np.array([
        [ -Xudot, 0 ,0], 
        [ 0, -Yvdot, -Yrdot], 
        [ 0, -Nvdot, -Nrdot] ],float)

    N_prime  = np.array([
        [ -Xu, 0 ,0], 
        [ 0, -Yv, -Yr], 
        [ 0, -Nv, -Nr] ],float)
 
    # Dimensional model (Fossen 2021, Appendix D)   
    T    = np.diag([1, 1, 1/L])
    Tinv = np.diag([1, 1, L])

    MA = (0.5 * rho * L**3) * Tinv @ Tinv @ \
        np.matmul( T, np.matmul( MA_prime,Tinv ) )
                  
    N =  (0.5 * rho * L**2 * U) * Tinv @ Tinv @ \
        np.matmul( T, np.matmul( N_prime, Tinv ) )
 
    M = MRB + MA            # system inertia matrix
    
    return M, N

 
 



#------------------------------------------------------------------------------

CYL_CD = np.array([                                        # MSS cylinderDrag.m: [Re, Cd] (DNV-RP-C205)
    [10211.0405297256, 1.20769], [15543.5423211490, 1.20369], [24434.8681540938, 1.20823],
    [35719.8807222993, 1.21060], [60880.7508493947, 1.21093], [86174.7436190610, 1.20901],
    [118086.431856986, 1.21134], [149262.255516420, 1.21148], [214708.876786124, 1.21171],
    [230940.445891927, 1.20109], [271581.385758833, 1.16918], [297098.185911873, 1.11163],
    [325252.018159382, 1.00926], [361942.848517458, 0.89411], [409798.156149477, 0.70855],
    [475302.392082697, 0.53155], [524734.960427919, 0.40785], [578921.928291835, 0.32683],
    [638292.879987135, 0.28422], [744046.366191920, 0.29711], [853237.628592033, 0.32280],
    [1068685.70450627, 0.38909], [1393381.37138896, 0.47033], [1831850.38455424, 0.53878],
    [2258395.50114555, 0.58372], [2899011.91287642, 0.62868], [3663279.03082383, 0.64803],
    [4937305.49552761, 0.67808]])
KAPPA_SUB = np.array([[2, 0.58], [5, 0.62], [10, 0.68], [20, 0.74], [40, 0.82], [50, 0.87], [100, 0.98]])
KAPPA_SUP = np.array([[2, 0.80], [5, 0.80], [10, 0.82], [20, 0.90], [40, 0.98], [50, 0.99], [100, 1.00]])

#------------------------------------------------------------------------------

def cylinderDrag(L, B, nu_r, re_length=None):
    """2-D drag coefficient of a cylinder in cross-flow, Cd(Re) times the aspect-ratio factor kappa(L/B)
    (MSS cylinderDrag.m, DNV-RP-C205): Re = U_cf B 1e6 with the diameter B and the cross-flow speed
    U_cf = sqrt(v_r^2 + w_r^2). re_length (not in MSS) replaces B in the Reynolds number; the MSS version before
    2026-10-05 used the length L. np.interp holds the end values, as the MSS code does."""
    U_cf = np.sqrt(nu_r[1] ** 2 + nu_r[2] ** 2)
    Re = U_cf * (B if re_length is None else re_length) * 1e6
    Cd = float(np.interp(Re, CYL_CD[:, 0], CYL_CD[:, 1]))
    tab = KAPPA_SUB if Re < 2e5 else KAPPA_SUP
    kappa = float(np.interp(L / B, tab[:, 0], tab[:, 1]))
    return Cd * kappa

#------------------------------------------------------------------------------

def coeffLiftDrag(b, S, CD_0, alpha, sigma=0.0, e=0.3):
    """Lift and drag coefficients of a submerged "wing" (MSS coeffLiftDrag.m, Beard and McLain 2012):
    AR = b^2/S, CL_alpha = pi AR / (1 + sqrt(1 + (AR/2)^2)), CL = (1 - sigma) CL_alpha alpha
    + sigma 2 sign(alpha) sin^2(alpha) cos(alpha), CD = CD_0 + CL^2 / (pi e AR), Oswald number e = 0.3."""
    AR = b**2 / S
    CL_alpha = np.pi * AR / (1 + np.sqrt(1 + (AR / 2) ** 2))
    alpha = np.asarray(alpha, float)
    CL = (1 - sigma) * CL_alpha * alpha + sigma * 2 * np.sign(alpha) * np.sin(alpha) ** 2 * np.cos(alpha)
    CD = CD_0 + CL**2 / (np.pi * e * AR)
    return CL, CD

#------------------------------------------------------------------------------

def spheroid(a, b, nu2, r_bG, rho=1025.0):
    """Rigid-body MRB and CRB(nu2) about the CO of a solid prolate spheroid with semi-axes a > b (MSS spheroid.m):
    m = 4/3 pi rho a b^2, Ix = 2/5 m b^2, Iy = Iz = 1/5 m (a^2 + b^2); about the CG
    MRB = diag(m, m, m, Ix, Iy, Iz), CRB = [m S(nu2), 0; 0, -S(Ig nu2)]; moved to the CO with H(r_bG)."""
    m = 4.0 / 3.0 * np.pi * rho * a * b**2
    Ix = 2.0 / 5.0 * m * b**2
    Iy = 1.0 / 5.0 * m * (a**2 + b**2)
    Ig = np.diag([Ix, Iy, Iy])
    MRB_CG = np.diag([m, m, m, Ix, Iy, Iy])
    CRB_CG = np.zeros((6, 6))
    CRB_CG[0:3, 0:3] = m * Smtrx(nu2)
    CRB_CG[3:6, 3:6] = -Smtrx(Ig @ np.asarray(nu2, float))
    H = Hmtrx(r_bG)
    return H.T @ MRB_CG @ H, H.T @ CRB_CG @ H

#------------------------------------------------------------------------------

def lamb_k(a, b):
    """Lamb's k-factors of a prolate spheroid (as in MSS imlay61.m): eccentricity e = sqrt(1 - (b/a)^2),
    k1 (surge), k2 (sway and heave) and k' (roll-free rotation about y and z). Returns (k1, k2, kp, e, alpha0, beta0)."""
    if a <= b:
        raise ValueError("a must be larger than b")
    e = np.sqrt(1 - (b / a) ** 2)
    lg = np.log((1 + e) / (1 - e))
    alpha0 = (2 * (1 - e**2) / e**3) * (0.5 * lg - e)
    beta0 = 1 / e**2 - (1 - e**2) / (2 * e**3) * lg
    k1 = alpha0 / (2 - alpha0)
    k2 = beta0 / (2 - beta0)
    kp = e**4 * (beta0 - alpha0) / ((2 - e**2) * (2 * e**2 - (2 - e**2) * (beta0 - alpha0)))
    return k1, k2, kp, e, alpha0, beta0

#------------------------------------------------------------------------------

def imlay61(a, b, nu, r44=0.0, rho=1026.0):
    """Added mass MA = diag(m k1, m k2, m k2, r44 Ix, k' Iy, k' Iy) and CA = m2c(MA, nu) of a prolate spheroid
    (MSS imlay61.m; Lamb 1932, Imlay 1961). m, Ix, Iy are the values of the displaced fluid."""
    if a < 0 or b < 0:
        raise ValueError("a and b must be positive")
    m = 4.0 / 3.0 * np.pi * rho * a * b**2
    Ix = 2.0 / 5.0 * m * b**2
    Iy = 1.0 / 5.0 * m * (a**2 + b**2)
    k1, k2, kp = lamb_k(a, b)[0:3]
    MA = np.diag([m * k1, m * k2, m * k2, r44 * Ix, kp * Iy, kp * Iy])
    return MA, m2c(MA, nu)

#------------------------------------------------------------------------------

def Dmtrx(T_126, zeta_45, MRB, MA, hydrostatics):
    """Diagonal linear damping matrix (MSS Dmtrx.m). Submerged vehicle: hydrostatics = [W, xg, yg, zg, xb, yb, zb]
    (W = B assumed, T3 = T2); surface craft: hydrostatics = 6 x 6 restoring matrix G (zeta3 = 0.2)."""
    M = MRB + MA
    T1, T2, T6 = T_126
    zeta4, zeta5 = zeta_45
    hs = np.asarray(hydrostatics, float)
    if hs.ndim == 1:
        W, r_bg, r_bb = hs[0], hs[1:4], hs[4:7]
        T3 = T2
        w4 = np.sqrt(W * (r_bg[2] - r_bb[2]) / M[3, 3])
        w5 = np.sqrt(W * (r_bg[2] - r_bb[2]) / M[4, 4])
        return np.diag([M[0, 0] / T1, M[1, 1] / T2, M[2, 2] / T3,
                        M[3, 3] * 2 * zeta4 * w4, M[4, 4] * 2 * zeta5 * w5, M[5, 5] / T6])
    w3, w4, w5 = (np.sqrt(hs[i, i] / M[i, i]) for i in (2, 3, 4))
    return np.diag([M[0, 0] / T1, M[1, 1] / T2, M[2, 2] * 2 * 0.2 * w3,
                    M[3, 3] * 2 * zeta4 * w4, M[4, 4] * 2 * zeta5 * w5, M[5, 5] / T6])

#------------------------------------------------------------------------------

def addedMassSurge(m, L, rho=1025.0):
    """Added mass in surge, A11 = -Xudot = 2.7 rho nabla^(5/3) / L^2 (Soding 1982; MSS addedMassSurge.m)."""
    return 2.7 * rho * (m / rho)**(5.0 / 3.0) / L**2
