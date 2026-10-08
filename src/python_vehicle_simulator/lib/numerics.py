#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numerical methods used by the MSS functions in Python: the integration steps of MSS (rk4.m, euler2.m, invQR.m)
and numpy versions of the MATLAB / Control System Toolbox functions that the MSS m-files call (round, interp1 with
'extrap', expm, c2d, lyap, care, lqr, ctrb, damp, ...).

Each function names the MATLAB or MSS function it follows (MSS: github.com/cybergalactic/MSS).
"""

import numpy as np
import math

#------------------------------------------------------------------------------

def mround(a):
    """MATLAB round (halves away from zero); Python's round() rounds halves to even."""
    return np.sign(a) * np.floor(np.abs(a) + 0.5)

#------------------------------------------------------------------------------

def euler2(xdot, x, h):
    """One Euler step (MSS euler2.m)."""
    return x + h * xdot

#------------------------------------------------------------------------------

def rk4(f, h, x, *args, **kwargs):
    """One 4th-order Runge-Kutta step x(k+1) for xdot = f(x, ...) (MSS rk4.m); f may return (xdot, ...)."""
    def F(z):
        out = f(z, *args, **kwargs)
        return out[0] if isinstance(out, tuple) else out
    k1 = F(x)
    k2 = F(x + 0.5 * h * k1)
    k3 = F(x + 0.5 * h * k2)
    k4 = F(x + h * k3)
    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

#------------------------------------------------------------------------------

def interp1_extrap(x, xp, fp):
    """MATLAB interp1(xp, fp, x, 'linear', 'extrap'): linear interpolation, linear extrapolation outside the table."""
    xp, fp = np.asarray(xp, float), np.asarray(fp, float)
    if x < xp[0]:
        return fp[0] + (x - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
    if x > xp[-1]:
        return fp[-1] + (x - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    return float(np.interp(x, xp, fp))

#------------------------------------------------------------------------------

def expm(A):
    """Matrix exponential (MATLAB expm) by scaling and squaring of a Taylor series, accurate to rounding for
    small matrices."""
    A = np.asarray(A, float)
    nrm = np.linalg.norm(A, 1)
    s = max(0, int(math.ceil(math.log2(nrm / 0.25))) if nrm > 0.25 else 0)
    As = A / 2.0**s
    E = np.eye(A.shape[0])
    term = np.eye(A.shape[0])
    for k in range(1, 40):
        term = term @ As / k
        E = E + term
        if np.linalg.norm(term, 1) < 1e-18 * np.linalg.norm(E, 1):
            break
    for _ in range(s):
        E = E @ E
    return E

#------------------------------------------------------------------------------

def c2d(A, B, h):
    """Zero-order hold: Phi = exp(A h), Delta = int_0^h exp(A s) ds B (MATLAB [Phi, Delta] = c2d(A, B, h))."""
    A = np.atleast_2d(np.asarray(A, float))
    B = np.asarray(B, float).reshape(A.shape[0], -1)
    n, m = B.shape
    Z = np.zeros((n + m, n + m))
    Z[:n, :n], Z[:n, n:] = A, B
    F = expm(Z * h)
    return F[:n, :n], F[:n, n:]

#------------------------------------------------------------------------------

def c2d_foh(A, B, C, D, h):
    """First-order hold (triangle hold) as in MATLAB and GNU Octave c2d(sys, h, 'foh'): with
    E = expm([A B 0; 0 0 I/h; 0 0 0] h), Phi = E11, G1 = E12, G2 = E13 the discrete model is
    a = Phi, b = G1 + Phi G2 - G2, c = C, d = D + C G2; its state is x - G2 u (not the continuous state x)."""
    A = np.atleast_2d(np.asarray(A, float))
    n = A.shape[0]
    B = np.asarray(B, float).reshape(n, -1)
    m = B.shape[1]
    C = np.atleast_2d(np.asarray(C, float))
    D = np.asarray(D, float).reshape(C.shape[0], m)
    Z = np.zeros((n + 2 * m, n + 2 * m))
    Z[:n, :n], Z[:n, n:n + m], Z[n:n + m, n + m:] = A, B, np.eye(m) / h
    E = expm(Z * h)
    Phi, G1, G2 = E[:n, :n], E[:n, n:n + m], E[:n, n + m:]
    return Phi, G1 + Phi @ G2 - G2, C, D + C @ G2

#------------------------------------------------------------------------------

def lyap(A, Q):
    """Solve A' P + P A + Q = 0 by the Kronecker product, small systems (MATLAB lyap(A', Q))."""
    A = np.atleast_2d(np.asarray(A, float))
    n = A.shape[0]
    Q = np.atleast_2d(np.asarray(Q, float))
    I = np.eye(n)
    P = np.linalg.solve(np.kron(I, A.T) + np.kron(A.T, I), -Q.reshape(-1, order="F")).reshape(n, n, order="F")
    return 0.5 * (P + P.T)

#------------------------------------------------------------------------------

def care(A, B, Q, R):
    """Stabilizing solution P = P' >= 0 (> 0 when (A, Q) is observable) of the algebraic Riccati
    equation P A + A' P - P B R^-1 B' P + Q = 0 from the stable eigenvectors of the Hamiltonian matrix, refined by
    three Newton-Kleinman steps (Lyapunov equations). Returns P and the largest residual relative to max|P|."""
    A = np.atleast_2d(np.asarray(A, float))
    n = A.shape[0]
    B = np.asarray(B, float).reshape(n, -1)
    Q = np.atleast_2d(np.asarray(Q, float))
    R = np.atleast_2d(np.asarray(R, float))
    Ri = np.linalg.inv(R)
    H = np.block([[A, -B @ Ri @ B.T], [-Q, -A.T]])
    w, V = np.linalg.eig(H)
    idx = np.where(w.real < 0)[0]
    if len(idx) != n:      # (A, B) not stabilizable or (A, Q) not detectable
        raise ValueError("no stabilizing solution")
    P = np.real(V[n:, idx] @ np.linalg.inv(V[:n, idx]))
    P = 0.5 * (P + P.T)
    for _ in range(3):
        K = Ri @ B.T @ P
        P = lyap(A - B @ K, Q + K.T @ R @ K)
    res = P @ A + A.T @ P - P @ B @ Ri @ B.T @ P + Q
    return P, np.abs(res).max() / np.abs(P).max()

#------------------------------------------------------------------------------

def lqr(A, B, Q, R):
    """LQ regulator as MATLAB [K, P, E] = lqr(A, B, Q, R): u = -K x with K = R^-1 B' P, E = eig(A - B K)."""
    P, _ = care(A, B, Q, R)
    B = np.asarray(B, float).reshape(P.shape[0], -1)
    K = np.linalg.inv(np.atleast_2d(R)) @ B.T @ P
    E = np.linalg.eigvals(np.atleast_2d(A) - B @ K)
    return K, P, E[np.lexsort((E.imag, E.real))]

#------------------------------------------------------------------------------

def ctrb(A, B):
    """Controllability matrix [B, A B, ..., A^(n-1) B]."""
    A = np.atleast_2d(A)
    B = np.asarray(B, float).reshape(A.shape[0], -1)
    return np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(A.shape[0])])

#------------------------------------------------------------------------------

def rank_sv(M, rtol=1e-10):
    """Rank and singular values; singular values below rtol times the largest one count as zero."""
    sv = np.linalg.svd(np.atleast_2d(M), compute_uv=False)
    return int(np.sum(sv > rtol * sv[0])), sv

#------------------------------------------------------------------------------

def damp(A):
    """MATLAB damp: eigenvalues (sorted by frequency), natural frequencies |lambda| and damping ratios
    -Re(lambda)/|lambda| (1 for a pole at the origin, as in MATLAB)."""
    e = np.linalg.eigvals(np.atleast_2d(A))
    wn = np.abs(e)
    zeta = np.where(wn > 0, -e.real / np.where(wn > 0, wn, 1.0), 1.0)
    idx = np.lexsort((e.imag, wn))
    return e[idx], wn[idx], zeta[idx]

#------------------------------------------------------------------------------

def tf_siso(A, B, C):
    """Numerator and denominator coefficients of c (sI - A)^-1 b (D = 0), from
    num(s) = det(sI - A + b c) - det(sI - A) (exact for a SISO system), and the zeros and poles."""
    A = np.atleast_2d(np.asarray(A, float))
    b = np.asarray(B, float).reshape(-1, 1)
    c = np.asarray(C, float).reshape(1, -1)
    den = np.poly(A)
    num = np.poly(A - b @ c) - den
    k = np.argmax(np.abs(num) > 1e-12 * np.abs(num).max())
    num = num[k:]
    return num, den, np.roots(num), np.roots(den)

#------------------------------------------------------------------------------

def invQR(A):
    """Matrix inverse by the QR decomposition (MSS invQR.m)."""
    A = np.asarray(A, float)
    if np.linalg.matrix_rank(A) < A.shape[0]:
        raise ValueError("The matrix is near-singular and not invertible.")
    Q, R = np.linalg.qr(A)
    return np.linalg.solve(R, Q.T)

#------------------------------------------------------------------------------

def vnorm(v):
    """Euclidean norm of a vector, sqrt(v'v) (as np.linalg.norm for a vector, without its overhead)."""
    return math.sqrt(float(v @ v))

#------------------------------------------------------------------------------

def expm_taylor(A):
    """Matrix exponential as in MSS expm_taylor.m: Taylor terms A^k/k! are added until the infinity norm of the last
    added term is at most 1e-6 (the m-file's stopping rule)."""
    A = np.asarray(A, float)
    n = A.shape[0]
    E = np.eye(n)
    term = np.eye(n)
    Ak = np.eye(n)
    k = 1
    while np.linalg.norm(term, np.inf) > 1e-6:
        Ak = Ak @ A
        term = Ak / math.factorial(k)
        E = E + term
        k += 1
    return E

#------------------------------------------------------------------------------

def obsv(A, H):
    """Observability matrix [H; H A; ...; H A^(n-1)]."""
    A = np.atleast_2d(A)
    H = np.atleast_2d(H)
    return np.vstack([H @ np.linalg.matrix_power(A, i) for i in range(A.shape[0])])

#------------------------------------------------------------------------------

def freqresp(A, B, C, D, w):
    """Frequency response C (j w I - A)^-1 B + D of a SISO state-space model at the frequencies w [rad/s]."""
    A = np.atleast_2d(np.asarray(A, float))
    n = A.shape[0]
    B = np.asarray(B, float).reshape(n)
    C = np.asarray(C, float).reshape(n)
    return np.array([C @ np.linalg.solve(1j * wk * np.eye(n) - A, B) + D for wk in np.atleast_1d(w)])
