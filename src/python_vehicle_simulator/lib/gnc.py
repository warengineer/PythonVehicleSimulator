#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNC functions. 

Reference: T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and
Motion Control. 2nd. Edition, Wiley. 
URL: www.fossen.biz/wiley

Author:     Thor I. Fossen
"""

import numpy as np
import math
from .numerics import vnorm

#------------------------------------------------------------------------------

def ssa(angle):
    """
    angle = ssa(angle) returns the smallest-signed angle in [ -pi, pi )
    """
    angle = (angle + math.pi) % (2 * math.pi) - math.pi
        
    return angle 

#------------------------------------------------------------------------------

def sat(x, x_min, x_max=None):
    """
    x = sat(x,x_min,x_max) saturates a signal x such that x_min <= x <= x_max
    x = sat(x,x_max) saturates symmetrically, -x_max <= x <= x_max (MSS sat.m)
    x can be a scalar or an array (element by element).
    """
    if x_max is None:
        x_max = x_min
        if np.any(np.asarray(x_max) <= 0):
            raise ValueError("x_max must contain positive elements")
        x_min = -x_max

    return np.minimum(np.maximum(x, x_min), x_max)

#------------------------------------------------------------------------------

def Smtrx(a):
    """
    S = Smtrx(a) computes the 3x3 vector skew-symmetric matrix S(a) = -S(a)'.
    The cross product satisfies: a x b = S(a)b. 
    """
 
    S = np.array([ 
        [ 0, -a[2], a[1] ],
        [ a[2],   0,     -a[0] ],
        [-a[1],   a[0],   0 ]  ])

    return S


#------------------------------------------------------------------------------

def Hmtrx(r):
    """
    H = Hmtrx(r) computes the 6x6 system transformation matrix
    H = [eye(3)     S'
         zeros(3,3) eye(3) ]       Property: inv(H(r)) = H(-r)

    If r = r_bg is the vector from the CO to the CG, the model matrices in CO and
    CG are related by: M_CO = H(r_bg)' * M_CG * H(r_bg). Generalized position and
    force satisfy: eta_CO = H(r_bg)' * eta_CG and tau_CO = H(r_bg)' * tau_CG 
    """

    H = np.identity(6,float)
    H[0:3, 3:6] = Smtrx(r).T

    return H

#------------------------------------------------------------------------------

def Rzyx(phi,theta,psi):
    """
    R = Rzyx(phi,theta,psi) computes the Euler angle rotation matrix R in SO(3)
    using the zyx convention
    """
    
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    cth  = math.cos(theta)
    sth  = math.sin(theta)
    cpsi = math.cos(psi)
    spsi = math.sin(psi)
    
    R = np.array([
        [ cpsi*cth, -spsi*cphi+cpsi*sth*sphi, spsi*sphi+cpsi*cphi*sth ],
        [ spsi*cth,  cpsi*cphi+sphi*sth*spsi, -cpsi*sphi+sth*spsi*cphi ],
        [ -sth,      cth*sphi,                 cth*cphi ] ])

    return R

#------------------------------------------------------------------------------

def Tzyx(phi,theta):
    """
    T = Tzyx(phi,theta) computes the Euler angle attitude
    transformation matrix T using the zyx convention
    """
    
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    cth  = math.cos(theta)
    sth  = math.sin(theta)    

    try: 
        T = np.array([
            [ 1,  sphi*sth/cth,  cphi*sth/cth ],
            [ 0,  cphi,          -sphi],
            [ 0,  sphi/cth,      cphi/cth] ])
        
    except ZeroDivisionError:  
        print ("Tzyx is singular for theta = +-90 degrees." )
        
    return T

#------------------------------------------------------------------------------

def attitudeEuler(eta,nu,sampleTime):
    """
    eta = attitudeEuler(eta,nu,sampleTime) computes the generalized 
    position/Euler angles eta[k+1]
    """
   
    p_dot   = np.matmul( Rzyx(eta[3], eta[4], eta[5]), nu[0:3] )
    v_dot   = np.matmul( Tzyx(eta[3], eta[4]), nu[3:6] )

    # Forward Euler integration
    eta[0:3] = eta[0:3] + sampleTime * p_dot
    eta[3:6] = eta[3:6] + sampleTime * v_dot

    return eta


#------------------------------------------------------------------------------

def m2c(M, nu):
    """
    C = m2c(M,nu) computes the Coriolis and centripetal matrix C from the
    mass matrix M and generalized velocity vector nu (Fossen 2021, Ch. 3)
    """

    M = 0.5 * (M + M.T)     # systematization of the inertia matrix

    if (len(nu) == 6):      #  6-DOF model
    
        M11 = M[0:3,0:3]
        M12 = M[0:3,3:6] 
        M21 = M12.T
        M22 = M[3:6,3:6] 
    
        nu1 = nu[0:3]
        nu2 = nu[3:6]
        dt_dnu1 = np.matmul(M11,nu1) + np.matmul(M12,nu2)
        dt_dnu2 = np.matmul(M21,nu1) + np.matmul(M22,nu2)

        #C  = [  zeros(3,3)      -Smtrx(dt_dnu1)
        #      -Smtrx(dt_dnu1)  -Smtrx(dt_dnu2) ]
        C = np.zeros( (6,6) )    
        C[0:3,3:6] = -Smtrx(dt_dnu1)
        C[3:6,0:3] = -Smtrx(dt_dnu1)
        C[3:6,3:6] = -Smtrx(dt_dnu2)
            
    else:   # 3-DOF model (surge, sway and yaw)
        #C = [ 0             0            -M(2,2)*nu(2)-M(2,3)*nu(3)
        #      0             0             M(1,1)*nu(1)
        #      M(2,2)*nu(2)+M(2,3)*nu(3)  -M(1,1)*nu(1)          0  ]    
        C = np.zeros( (3,3) ) 
        C[0,2] = -M[1,1] * nu[1] - M[1,2] * nu[2]
        C[1,2] =  M[0,0] * nu[0] 
        C[2,0] = -C[0,2]       
        C[2,1] = -C[1,2]
        
    return C

#------------------------------------------------------------------------------

# Hoerner's curve [B/2T, C_D], the digitized data of MSS Hoerner.m (Leite et al. 1998)
HOERNER = np.array([
    [0.0108623, 1.96608], [0.176606, 1.96573], [0.353025, 1.89756], [0.451863, 1.78718], [0.472838, 1.58374],
    [0.492877, 1.27862], [0.493252, 1.21082], [0.558473, 1.08356], [0.646401, 0.998631], [0.833589, 0.87959],
    [0.988002, 0.828415], [1.30807, 0.759941], [1.63918, 0.691442], [1.85998, 0.657076], [2.31288, 0.630693],
    [2.59998, 0.596186], [3.00877, 0.586846], [3.45075, 0.585909], [3.7379, 0.559877], [4.00309, 0.559315]])

def Hoerner(B,T):
    """
    CY_2D = Hoerner(B,T)
    Hoerner computes the 2D Hoerner cross-flow form coeff. as a function of beam
    B and draft T.The data is digitized and interpolation is used to compute
    other data point than those in the table. Same data as MSS Hoerner.m; a ratio
    B/(2T) outside the table gets the value at the nearest end.
    """

    CY_2D = np.interp( B / (2 * T), HOERNER[:,0], HOERNER[:,1] )

    return CY_2D

#------------------------------------------------------------------------------

def crossFlowDrag(L,B,T,nu_r,drag_model="Hoerner",n_strips=20,rho=1025.0,
                  re_length=None):
    """
    tau_crossflow = crossFlowDrag(L,B,T,nu_r,drag_model) computes the
    cross-flow drag integrals for a marine craft using strip theory
    (MSS crossFlowDrag.m).

    M d/dt nu_r + C(nu_r)*nu_r + D*nu_r + g(eta) = tau + tau_crossflow

    The strips are evaluated at their midpoints:
        Y = -sum 1/2 rho T Cd |v_r + x r|(v_r + x r) dx,  N = -sum x (...),
    and the heave force Z and pitch moment M the same with w_r + x q.
    Returns tau_crossflow = [0, Y, Z, 0, M, N].

    Inputs:
        L, B, T:    length, beam and draft (m)
        nu_r:       relative velocity [u-u_c, v-v_c, w-w_c, p, q, r]
        drag_model: 'Hoerner' (2D coefficient of Hoerner's curve, default) or
                    'cylinder' (cylinderDrag, for cylinder-shaped AUVs)
        n_strips:   number of strips (20 in MSS)
        rho:        density of water (1025 kg/m^3 in MSS)
        re_length:  length in the Reynolds number of cylinderDrag instead of
                    the diameter B (not in MSS; for comparisons)
    """
    if drag_model == "Hoerner":
        Cd_2D = Hoerner(B,T)    # 2D drag coefficient based on Hoerner's curve
    elif drag_model == "cylinder":
        from python_vehicle_simulator.lib.models import cylinderDrag
        Cd_2D = cylinderDrag(L, B, nu_r, re_length)
    else:
        raise ValueError(f"unsupported drag model {drag_model}")

    dx = L / n_strips
    v_r, w_r, q, r = nu_r[1], nu_r[2], nu_r[4], nu_r[5]
    xL = -L / 2 + (np.arange(1, n_strips + 1) - 0.5) * dx      # strip midpoints
    U_h = np.abs(v_r + xL * r) * (v_r + xL * r)
    U_v = np.abs(w_r + xL * q) * (w_r + xL * q)
    k = 0.5 * rho * T * Cd_2D * dx

    tau_crossflow = np.array([0.0, -k * U_h.sum(), -k * U_v.sum(), 0.0,
                              -k * (xL * U_v).sum(), -k * (xL * U_h).sum()])

    return tau_crossflow

#------------------------------------------------------------------------------

def forceLiftDrag(b,S,CD_0,alpha,U_r,rho=1026.0):
    """
    tau_liftdrag = forceLiftDrag(b,S,CD_0,alpha,Ur) computes the hydrodynamic
    lift and drag forces of a submerged "wing profile" for varying angle of
    attack (Beard and McLain 2012). Application:
    
      M d/dt nu_r + C(nu_r)*nu_r + D*nu_r + g(eta) = tau + tau_liftdrag
    
    Inputs:
        b:     wing span (m)
        S:     wing area (m^2)
        CD_0:  parasitic drag (alpha = 0), typically 0.1-0.2 for a streamlined body
        alpha: angle of attack, scalar or vector (rad)
        U_r:   relative speed (m/s)

    Returns:
        tau_liftdrag:  6x1 generalized force vector

    The coefficients come from models.coeffLiftDrag (MSS coeffLiftDrag.m, Oswald
    efficiency number e = 0.3) with sigma = 0 (linear lift), as in MSS
    forceLiftDrag.m; rho = 1026 kg/m^3 as in MSS.
    """
    from python_vehicle_simulator.lib.models import coeffLiftDrag

    CL, CD = coeffLiftDrag(b, S, CD_0, alpha, 0.0)
    F_drag = 0.5 * rho * U_r**2 * S * CD    # drag force
    F_lift = 0.5 * rho * U_r**2 * S * CL    # lift force

    # transform from FLOW axes to BODY axes using angle of attack
    tau_liftdrag = np.array([
        np.cos(alpha) * (-F_drag) - np.sin(alpha) * (-F_lift),
        0.0,
        np.sin(alpha) * (-F_drag) + np.cos(alpha) * (-F_lift),
        0.0,
        0.0,
        0.0 ])

    return tau_liftdrag
    
#------------------------------------------------------------------------------

def gvect(W,B,theta,phi,r_bg,r_bb):
    """
    g = gvect(W,B,theta,phi,r_bg,r_bb) computes the 6x1 vector of restoring 
    forces about an arbitrarily point CO for a submerged body. 
    
    Inputs:
        W, B: weight and buoyancy (kg)
        phi,theta: roll and pitch angles (rad)
        r_bg = [x_g y_g z_g]: location of the CG with respect to the CO (m)
        r_bb = [x_b y_b z_b]: location of the CB with respect to th CO (m)
        
    Returns:
        g: 6x1 vector of restoring forces about CO
    """

    sth  = math.sin(theta)
    cth  = math.cos(theta)
    sphi = math.sin(phi)
    cphi = math.cos(phi)

    g = np.array([
        (W-B) * sth,
        -(W-B) * cth * sphi,
        -(W-B) * cth * cphi,
        -(r_bg[1]*W-r_bb[1]*B) * cth * cphi + (r_bg[2]*W-r_bb[2]*B) * cth * sphi,
        (r_bg[2]*W-r_bb[2]*B) * sth         + (r_bg[0]*W-r_bb[0]*B) * cth * cphi,
        -(r_bg[0]*W-r_bb[0]*B) * cth * sphi - (r_bg[1]*W-r_bb[1]*B) * sth      
        ])
    
    return g

#------------------------------------------------------------------------------

D2R = np.pi / 180.0      # degrees to radians (MSS conversion.m)

#------------------------------------------------------------------------------

def eulerang(phi, theta, psi):
    """J = blockdiag(Rzyx, Tzyx) and the two blocks (MSS eulerang.m)."""
    R, T = Rzyx(phi, theta, psi), Tzyx(phi, theta)
    J = np.zeros((6, 6))
    J[0:3, 0:3], J[3:6, 3:6] = R, T
    return J, R, T

#------------------------------------------------------------------------------

def Rquat(q):
    """Rotation matrix of the unit quaternion q = [eta, eps1, eps2, eps3] (MSS Rquat.m)."""
    q = np.asarray(q, float)
    if abs(np.linalg.norm(q) - 1) > 1e-6:
        raise ValueError("norm(q) must be equal to 1")
    S = Smtrx(q[1:4])
    return np.eye(3) + 2 * q[0] * S + 2 * S @ S

#------------------------------------------------------------------------------

def Tquat(u):
    """4 x 3 matrix Tq with q_dot = Tq(q) w for a quaternion u = q, or the 4 x 4 matrix Tw with q_dot = Tw(w) q for
    an angular velocity u = w (MSS Tquat.m)."""
    u = np.asarray(u, float)
    if len(u) == 4:
        eta, e1, e2, e3 = u
        return 0.5 * np.array([[-e1, -e2, -e3], [eta, -e3, e2], [e3, eta, -e1], [-e2, e1, eta]])
    w = u[0:3]
    T = np.zeros((4, 4))
    T[0, 1:4] = -w
    T[1:4, 0] = w
    T[1:4, 1:4] = -Smtrx(w)
    return 0.5 * T

#------------------------------------------------------------------------------

def quatern(q):
    """J = blockdiag(Rquat, Tquat) of size 7 x 6 and the two blocks (MSS quatern.m)."""
    R, T = Rquat(q), Tquat(q)
    J = np.zeros((7, 6))
    J[0:3, 0:3], J[3:7, 3:6] = R, T
    return J, R, T

#------------------------------------------------------------------------------

def gRvect(W, B, R, r_bG, r_bB):
    """Restoring forces of a submerged body from the rotation matrix R (Euler angles or quaternion) (MSS gRvect.m)."""
    xg, yg, zg = r_bG
    xb, yb, zb = r_bB
    return np.array([-(W - B) * R[2, 0], -(W - B) * R[2, 1], -(W - B) * R[2, 2],
                     -(yg * W - yb * B) * R[2, 2] + (zg * W - zb * B) * R[2, 1],
                     -(zg * W - zb * B) * R[2, 0] + (xg * W - xb * B) * R[2, 2],
                     -(xg * W - xb * B) * R[2, 1] + (yg * W - yb * B) * R[2, 0]])

#------------------------------------------------------------------------------

def Rz(psi):
    """Rotation matrix R(psi) of the 3-DOF horizontal-plane model, eta = [N, E, psi] (= Rzyx(0, 0, psi))."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

#------------------------------------------------------------------------------

def S3(r):
    """Skew-symmetric matrix of the 3-DOF model, dR(psi)/dt = R(psi) S3(r)."""
    return np.array([[0.0, -r, 0.0], [r, 0.0, 0.0], [0.0, 0.0, 0.0]])

#------------------------------------------------------------------------------

G_MSS = 9.81      # acceleration of gravity of the MSS ship models (m/s^2)

#------------------------------------------------------------------------------

def quatprod(q1, q2):
    """Quaternion (Hamilton) product q1 x q2 for scalar-first quaternions (MSS quatprod.m)."""
    eta1, eps1 = q1[0], np.asarray(q1[1:4], float)
    eta2, eps2 = q2[0], np.asarray(q2[1:4], float)
    return np.concatenate(([eta1 * eta2 - eps1 @ eps2], eta2 * eps1 + eta1 * eps2 + np.cross(eps1, eps2)))

#------------------------------------------------------------------------------

def euler2q(phi, theta, psi):
    """Unit quaternion from Euler angles (MSS euler2q.m; like the m-file it divides by q'q, which is 1)."""
    cy, sy = math.cos(psi * 0.5), math.sin(psi * 0.5)
    cp, sp = math.cos(theta * 0.5), math.sin(theta * 0.5)
    cr, sr = math.cos(phi * 0.5), math.sin(phi * 0.5)
    q = np.array([cy * cp * cr + sy * sp * sr,
                  cy * cp * sr - sy * sp * cr,
                  sy * cp * sr + cy * sp * cr,
                  sy * cp * cr - cy * sp * sr])
    return q / (q @ q)

#------------------------------------------------------------------------------

def q2euler(q):
    """Euler angles (phi, theta, psi) from a unit quaternion (MSS q2euler.m)."""
    q = np.asarray(q, float)
    q = q / vnorm(q)
    R = Rquat(q)
    phi = math.atan2(R[2, 1], R[2, 2])
    r31 = R[2, 0]
    if abs(r31) > 1:
        r31 = math.copysign(1.0, r31)
    theta = -math.asin(r31)
    psi = math.atan2(R[1, 0], R[0, 0])
    return phi, theta, psi

#------------------------------------------------------------------------------

def R2euler(R):
    """Euler angles from a rotation matrix (MSS R2euler.m)."""
    return math.atan2(R[2, 1], R[2, 2]), -math.asin(R[2, 0]), math.atan2(R[1, 0], R[0, 0])

#------------------------------------------------------------------------------

def expq(w, h):
    """exp(Tquat(w) h) in closed form, cos(a) I + sin(a)/|w| (2 Tquat(w)) with a = |w| h/2
    (the exponential map used by quatObserver.m, quatMEKF.m and ins_mekf.m)."""
    w = np.asarray(w, float)
    nw = math.sqrt(w @ w)
    a = 0.5 * nw * h
    T2 = 2.0 * Tquat(w)
    if nw < 1e-300:
        return np.eye(4)
    return math.cos(a) * np.eye(4) + (math.sin(a) / nw) * T2

#------------------------------------------------------------------------------

def satlim(x, x_min, x_max):
    """Asymmetric saturation x_min <= x <= x_max (MSS satlim.m)."""
    if np.any(np.asarray(x_min) >= np.asarray(x_max)):
        raise ValueError("Each element of x_min must be strictly smaller than the element of x_max.")
    if np.isscalar(x) and np.isscalar(x_min) and np.isscalar(x_max):
        return min(max(x, x_min), x_max)
    return np.minimum(np.maximum(x, x_min), x_max)
