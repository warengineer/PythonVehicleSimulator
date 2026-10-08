#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thrust configuration and control allocation (MSS thrConfig.m, allocPseudoinverse.m, ...).

Each function names its MSS m-file (MSS: github.com/cybergalactic/MSS).
"""

import numpy as np
import math
from .numerics import invQR
from .gnc import D2R, sat

#------------------------------------------------------------------------------

def thrConfig(alpha, l_x, l_y):
    """Thrust configuration matrix T(alpha), 3 x r (MSS thrConfig.m): 'M' main propeller [1, 0, -l_y], 'T' tunnel
    thruster [0, 1, l_x], a number = azimuth angle a: [cos a, sin a, l_x sin a - l_y cos a]."""
    T = np.zeros((3, len(alpha)))
    for i, a in enumerate(alpha):
        if isinstance(a, str) and a == "M":
            T[:, i] = [1.0, 0.0, -l_y[i]]
        elif isinstance(a, str) and a == "T":
            T[:, i] = [0.0, 1.0, l_x[i]]
        else:
            T[:, i] = [np.cos(a), np.sin(a), l_x[i] * np.sin(a) - l_y[i] * np.cos(a)]
    return T

#------------------------------------------------------------------------------

def allocPseudoinverse(K, T, W, tau):
    """Unconstrained allocation u = K^-1 W^-1 T' (T W^-1 T')^-1 tau (MSS allocPseudoinverse.m) for diagonal K, W."""
    K, W = np.asarray(K, float), np.asarray(W, float)
    Winv, Kinv = np.diag(1 / np.diag(W)), np.diag(1 / np.diag(K))
    return Kinv @ Winv @ T.T @ invQR(T @ Winv @ T.T) @ np.asarray(tau, float)

#------------------------------------------------------------------------------

class QPError(RuntimeError):
    pass

#------------------------------------------------------------------------------

def qp_box_eq(H, g, A, b, lb, ub, x0, max_iter=200, tol=1e-10):
    """Primal active-set method (Nocedal and Wright 2006, Algorithm 16.3) for the convex QP
        min 1/2 x'Hx + g'x   s.t.   A x = b,   lb <= x <= ub     (H > 0; entries of lb, ub may be -inf, +inf).
    x0 must be feasible (a warm start). The working set holds the bounds that are fixed; each iteration solves the
    equality-constrained subproblem (EQP) for the free variables, steps to the first blocking bound or, at the minimizer
    of the EQP, drops the bound with the most negative multiplier. A full step reaches the EQP minimizer, so the next
    iteration goes straight to the multiplier test (what is left of the step is rounding noise in badly scaled
    problems); a bound that is dropped and blocks the very next step at once ends the iteration. Returns x and a dict
    (iterations, 'lower'/'upper' index lists, multipliers nu of A x = b, mu of the bounds; mu is returned as zero when
    the iteration ends at a rounding-level multiplier). Raises QPError if it has not converged after max_iter
    iterations."""
    H, g, A, b = (np.asarray(v, float) for v in (H, g, A, b))
    lb, ub, x = (np.array(v, float) for v in (lb, ub, x0))
    n, m = len(x), A.shape[0]
    scale = 1.0 + np.max(np.abs(x))
    if np.any(x < lb - 1e-9 * scale) or np.any(x > ub + 1e-9 * scale) or \
            np.max(np.abs(A @ x - b), initial=0.0) > 1e-9 * (1.0 + np.max(np.abs(b))):
        raise QPError("qp_box_eq: x0 is not feasible")
    x = np.clip(x, lb, ub)
    at_lo = np.isfinite(lb) & (x <= lb)
    at_hi = np.isfinite(ub) & (x >= ub) & ~at_lo
    full_step, released = False, -1

    def done(it, nu, mu):
        return x, dict(iterations=it, lower=[int(i) for i in np.nonzero(at_lo)[0]],
                       upper=[int(i) for i in np.nonzero(at_hi)[0]], nu=nu, mu=mu)

    for it in range(1, max_iter + 1):
        fixed = at_lo | at_hi
        F = np.nonzero(~fixed)[0]
        c = H @ x + g
        nF = len(F)
        KKT = np.zeros((nF + m, nF + m))
        KKT[:nF, :nF] = H[np.ix_(F, F)]
        KKT[:nF, nF:] = A[:, F].T
        KKT[nF:, :nF] = A[:, F]
        sol = np.linalg.solve(KKT, np.r_[-c[F], np.zeros(m)])
        p = np.zeros(n)
        p[F] = sol[:nF]
        nu = sol[nF:]
        if full_step or np.all(np.abs(p) <= tol * (1.0 + np.abs(x))):
            grad = c + A.T @ nu                         # stationarity: grad_i = mu_i (lower), -mu_i (upper)
            mu = np.where(at_lo, grad, np.where(at_hi, -grad, 0.0))
            gscale = 1.0 + np.abs(c) + np.abs(A.T) @ np.abs(nu)   # size of the terms that form grad_i
            if np.all(mu[fixed] >= -1e-9 * gscale[fixed]):
                return done(it, nu, mu)
            j = int(np.argmin(np.where(fixed, mu / gscale, np.inf)))
            at_lo[j] = at_hi[j] = False
            full_step, released = False, j
            continue
        alpha, block, side = 1.0, -1, 0
        for i in F:
            if p[i] < 0 and np.isfinite(lb[i]):
                a = (lb[i] - x[i]) / p[i]
                if a < alpha:
                    alpha, block, side = a, i, -1
            elif p[i] > 0 and np.isfinite(ub[i]):
                a = (ub[i] - x[i]) / p[i]
                if a < alpha:
                    alpha, block, side = a, i, 1
        if block >= 0 and block == released and alpha <= 1e-12:
            (at_lo if side < 0 else at_hi)[block] = True        # a rounding-level multiplier: stop here
            return done(it, nu, np.zeros(n))
        released = -1
        x = x + max(alpha, 0.0) * p
        if block >= 0:
            if side < 0:
                x[block], at_lo[block] = lb[block], True
            else:
                x[block], at_hi[block] = ub[block], True
            full_step = False
        else:
            full_step = True
    raise QPError(f"qp_box_eq: no convergence in {max_iter} iterations")

#------------------------------------------------------------------------------

def ginv(T, W=None):
    """Generalized inverse T_W^+ = W^-1 T' (T W^-1 T')^-1 (W = I: Moore-Penrose)."""
    T = np.asarray(T, float)
    Winv = np.eye(T.shape[1]) if W is None else np.diag(1.0 / np.diag(W))
    return Winv @ T.T @ np.linalg.inv(T @ Winv @ T.T)

#------------------------------------------------------------------------------

def alloc_clip(T, W, tau, f_max):
    """Generalized inverse, then each force clipped to [-f_max, f_max] (what a saturating actuator does)."""
    return sat(ginv(T, W) @ tau, f_max)

#------------------------------------------------------------------------------

def alloc_scaled(T, W, tau, f_max):
    """Generalized inverse scaled down until the largest force ratio |f_i|/f_max_i is 1 (keeps the direction of tau)."""
    f = ginv(T, W) @ tau
    k = np.max(np.abs(f) / f_max)
    return f / k if k > 1.0 else f

#------------------------------------------------------------------------------

def alloc_rpi(T, W, tau, f_max):
    """Redistributed pseudo-inverse: saturate the thrusters that exceed their limits, fix them, and allocate the
    remaining generalized force to the free thrusters again; repeat. Every pass fixes at least one thruster, so it ends
    after at most r passes; when the free thrusters can no longer span tau (rank of their columns < n), the forces are
    clipped as they are. A heuristic: it need not deliver an attainable tau and is not optimal in general."""
    T = np.asarray(T, float)
    r = T.shape[1]
    Wd = np.ones(r) if W is None else np.diag(W)
    free = np.ones(r, bool)
    f = np.zeros(r)
    for _ in range(r):
        TF = T[:, free]
        if TF.shape[1] < T.shape[0] or np.linalg.matrix_rank(TF) < T.shape[0]:
            break
        fF = ginv(TF, np.diag(Wd[free])) @ (tau - T[:, ~free] @ f[~free])
        over = np.abs(fF) > f_max[free] * (1 + 1e-12)
        idx = np.nonzero(free)[0]
        f[idx] = fF
        if not over.any():
            return f
        f[idx[over]] = np.sign(fF[over]) * f_max[idx[over]]
        free[idx[over]] = False
    return sat(f, f_max)

#------------------------------------------------------------------------------

def alloc_qp(T, W, Q, tau, f_min, f_max, f0=None, **kw):
    """min f'Wf + s'Qs, T f = tau + s, f_min <= f <= f_max (eq. 12.255, beta = 0)"""
    T, tau = np.asarray(T, float), np.asarray(tau, float)
    n, r = T.shape
    H = 2.0 * np.block([[np.asarray(W, float), np.zeros((r, n))],
                        [np.zeros((n, r)), np.asarray(Q, float)]])
    A = np.hstack([T, -np.eye(n)])                    # T f - s = tau
    f0 = np.zeros(r) if f0 is None else np.clip(f0, f_min, f_max)
    z0 = np.r_[f0, T @ f0 - tau]                      # a feasible start
    lb = np.r_[f_min, -np.inf * np.ones(n)]           # the slack is free
    ub = np.r_[f_max, np.inf * np.ones(n)]
    z, info = qp_box_eq(H, np.zeros(r + n), A, tau, lb, ub, z0, **kw)
    return z[:r], z[r:], info

#------------------------------------------------------------------------------

def alloc_minmax(T, W, Q, beta, tau, f_min, f_max, tol=1e-6):
    """min f'Wf + s'Qs + beta fbar  s.t.  T f = tau + s, f_min <= f <= f_max, -fbar <= f_i <= fbar (eq. 12.255).
    For a fixed fbar = c the problem is alloc_qp with the bounds tightened to [max(f_min, -c), min(f_max, c)]; its
    optimal value plus beta c is convex in c, so a golden-section search over c finds the optimum.
    Returns f, s, fbar."""
    def J(c):
        f, s, _ = alloc_qp(T, W, Q, tau, np.maximum(f_min, -c), np.minimum(f_max, c))
        return f @ np.asarray(W, float) @ f + s @ np.asarray(Q, float) @ s + beta * c, f, s
    a, b = 0.0, float(np.max(np.maximum(np.abs(f_min), np.abs(f_max))))
    gr = (np.sqrt(5.0) - 1) / 2
    c1, c2 = b - gr * (b - a), a + gr * (b - a)
    J1, J2 = J(c1)[0], J(c2)[0]
    while b - a > tol * (1.0 + b):
        if J1 < J2:
            b, c2, J2 = c2, c1, J1
            c1 = b - gr * (b - a)
            J1 = J(c1)[0]
        else:
            a, c1, J1 = c1, c2, J2
            c2 = a + gr * (b - a)
            J2 = J(c2)[0]
    c = 0.5 * (a + b)
    _, f, s = J(c)
    return f, s, c

#------------------------------------------------------------------------------

def attainable_scale(T, tau, f_min, f_max):
    """Largest lam >= 0 with lam tau in the attainable set Phi = {T f : f_min <= f <= f_max} of a 3 x r matrix T
    (Durham's direct method scales tau to the boundary of Phi). Phi is a zonotope: its facets are normal to the cross
    products d = t_i x t_j of pairs of columns, and lam = min over facets with d'tau > 0 of h(d) / d'tau, where
    h(d) = sum_k max(d't_k f_max_k, d't_k f_min_k) is the support function. Returns lam (np.inf if tau = 0)."""
    T, tau = np.asarray(T, float), np.asarray(tau, float)
    r = T.shape[1]
    lam = np.inf
    for i in range(r):
        for j in range(i + 1, r):
            d = np.cross(T[:, i], T[:, j])
            nd = np.linalg.norm(d)
            if nd < 1e-12 * np.linalg.norm(T[:, i]) * np.linalg.norm(T[:, j]):
                continue
            for dd in (d / nd, -d / nd):
                dt = dd @ tau
                if dt > 0:
                    proj = dd @ T
                    h = np.sum(np.maximum(proj * f_max, proj * f_min))
                    lam = min(lam, h / dt)
    return lam

#------------------------------------------------------------------------------

def T_extended(kinds, l_x, l_y):
    """Extended thrust configuration matrix: 'T' tunnel [0, 1, l_x], 'M' main [1, 0, -l_y], 'A' azimuth -> two columns
    [1, 0, -l_y] (F_x) and [0, 1, l_x] (F_y). Constant (no angles)."""
    cols = []
    for k, lx, ly in zip(kinds, l_x, l_y):
        if k == "T":
            cols.append([0.0, 1.0, lx])
        elif k == "M":
            cols.append([1.0, 0.0, -ly])
        else:
            cols += [[1.0, 0.0, -ly], [0.0, 1.0, lx]]
    return np.array(cols).T

#------------------------------------------------------------------------------

def alloc_extended(tau, kinds, l_x, l_y, K, W=None):
    """Unconstrained allocation with virtual forces: f_e = T_e,W^+ tau, then each azimuth pair (F_x, F_y) gives the thrust
    F = sqrt(F_x^2 + F_y^2) >= 0 and the angle alpha = atan2(F_y, F_x). K = diag of the force coefficients of the r real
    thrusters (u = F / K). Returns u (r), alpha (one per azimuth), f_e and the coefficients K_e of the virtual forces."""
    Te = T_extended(kinds, l_x, l_y)
    Kd = np.diag(K)
    Ke = np.concatenate([[k] if kd != "A" else [k, k] for kd, k in zip(kinds, Kd)])
    We = None if W is None else np.diag(np.concatenate(
        [[w] if kd != "A" else [w, w] for kd, w in zip(kinds, np.diag(W))]))
    fe = ginv(Te, None if We is None else We) @ tau
    u, alpha, j = [], [], 0
    for kd, k in zip(kinds, Kd):
        if kd == "A":
            u.append(np.hypot(fe[j], fe[j + 1]) / k)
            alpha.append(np.arctan2(fe[j + 1], fe[j]))
            j += 2
        else:
            u.append(fe[j] / k)
            j += 1
    return np.array(u), np.array(alpha), fe, Ke

#------------------------------------------------------------------------------

# SIMosv.m: allocation model (its own thruster data), DP controller and scenario
SIMOSV = dict(K_max=np.diag([300e3, 300e3, 655e3, 655e3]), n_max=np.array([140.0, 140.0, 150.0, 150.0]),
              l_x=np.array([37.0, 35.0, -42.0, -42.0]), l_y=np.array([0.0, 0.0, 7.0, -7.0]),
              alpha0=np.array([-28.0, 28.0]) * D2R, az_max=60 * D2R, T_final=250.0, h=0.05,
              Vc=0.5, betaVc=-140 * D2R, wn=0.1 * np.diag([1.0, 1.0, 3.0]), zeta=np.eye(3), T_f=30.0,
              eta0=np.array([5.0, 5.0, 0.0, 5 * D2R, 2 * D2R, 0.0]), psi_ref2=40 * D2R, t_turn=50.0,
              w=(1.0, 100.0, 1.0, 0.1), max_rate_alpha=0.3, max_rate_u=0.1)


#------------------------------------------------------------------------------

class OSVAlloc:
    """The criterion of optimalAlloc in SIMosv.m (solved there by fmincon/SQP to convergence, MATLAB only),
        min  w1 |u|^2 + w2 |s|^2 + w3 |alpha - alpha_old|^2 + w4 |u - u_old|^2
        s.t. T(alpha) K u - tau + s = 0,  -az_max <= alpha <= az_max,  -1 <= u <= 1,
             |alpha - alpha_old| <= 0.3 h,  |u - u_old| <= 0.1 h       (rates 0.3 rad/s and 0.1 per second),
    u = |n/n_max| n/n_max, (w1, w2, w3, w4) = (1, 100, 1, 0.1). Here the slack is measured in kN
    (s_unit = 1e3; SIMosv.m measures it in N), which keeps the weights within six orders of magnitude, and the problem
    is solved as ONE local convex QP per sample (Johansen, Fossen and Berge 2004; Fossen 2011, eq. 12.263): the
    constraint is linearized at the previous solution, T(alpha) K u ~ T(alpha_old) K u
    + d(T(alpha) K u_old)/dalpha (alpha - alpha_old), which is linear in (alpha, u). qp_box_eq solves it in
    x = [alpha, u, z], z = sqrt(w2) s/s_unit (the slack scaled to kN and by the square root of its weight).
    Returns alpha, u, the slack s = tau - T(alpha) K u (N) of the true model, and info."""

    def __init__(self, data=SIMOSV, s_unit=1e3):
        self.d = data
        self.w = data["w"]
        self.K = np.asarray(data["K_max"], float)
        self.k = np.diag(self.K)
        self.su = s_unit

    def T(self, alpha):
        return thrConfig(["T", "T", alpha[0], alpha[1]], self.d["l_x"], self.d["l_y"])

    def jac_alpha(self, alpha, u):
        """Jacobian of T(alpha) K u with respect to (alpha1, alpha2)."""
        lx, ly, K = self.d["l_x"], self.d["l_y"], self.k
        J = np.zeros((3, 2))
        for j, i in enumerate((2, 3)):
            a = alpha[j]
            J[:, j] = K[i] * u[i] * np.array([-np.sin(a), np.cos(a), lx[i] * np.cos(a) + ly[i] * np.sin(a)])
        return J

    def bounds(self, alpha_old, u_old, h):
        d = self.d
        ra, ru = d["max_rate_alpha"] * h, d["max_rate_u"] * h
        lo = np.r_[np.maximum(-d["az_max"], alpha_old - ra), np.maximum(-1.0, u_old - ru)]
        hi = np.r_[np.minimum(d["az_max"], alpha_old + ra), np.minimum(1.0, u_old + ru)]
        return lo, hi

    def local_qp(self, tau, alpha_old, u_old, a_lin, u_lin):
        """H, g, A, b of the local QP in x = [alpha, u, z], linearized at
        (a_lin, u_lin); the rows of A x = b are in units of s_unit."""
        w1, w2, w3, w4 = self.w
        TK = self.T(a_lin) @ self.K
        Ja = self.jac_alpha(a_lin, u_lin)
        H = 2.0 * np.diag(np.r_[w3, w3, (w1 + w4) * np.ones(4), np.ones(3)])
        g = np.r_[-2 * w3 * alpha_old, -2 * w4 * u_old, np.zeros(3)]
        A = np.hstack([Ja, TK, self.su * np.eye(3) / np.sqrt(w2)]) / self.su
        b = (tau + Ja @ a_lin) / self.su
        return H, g, A, b, TK, Ja

    def __call__(self, tau, alpha_old, u_old, h):
        tau = np.asarray(tau, float)
        w1, w2, w3, w4 = self.w
        lo, hi = self.bounds(alpha_old, u_old, h)
        xa = np.clip(np.r_[alpha_old, u_old], lo, hi)
        H, g, A, b, TK, Ja = self.local_qp(tau, alpha_old, u_old, xa[0:2], xa[2:6])
        z0 = np.sqrt(w2) * (tau - TK @ xa[2:6]) / self.su        # makes x0 = [xa, z0] feasible
        lb = np.r_[lo, -np.inf * np.ones(3)]
        ub = np.r_[hi, np.inf * np.ones(3)]
        x, info = qp_box_eq(H, g, A, b, lb, ub, np.r_[xa, z0], tol=1e-13)
        its = info["iterations"]
        a, u = x[0:2], x[2:6]
        s = tau - self.T(a) @ self.K @ u
        sk = s / self.su
        cost = w1 * u @ u + w2 * sk @ sk + w3 * (a - alpha_old) @ (a - alpha_old) + w4 * (u - u_old) @ (u - u_old)
        return a, u, s, dict(iterations=its, cost=cost, s_lin=x[6:9] * self.su / np.sqrt(w2))

#------------------------------------------------------------------------------

def dls(A, b, lam):
    """Damped least squares x = A'(A A' + lam I)^-1 b (lam = 0 gives the
    minimum-norm solution; lam > 0 keeps x bounded near singularities)."""
    return A.T @ np.linalg.solve(A @ A.T + lam * np.eye(A.shape[0]), b)
