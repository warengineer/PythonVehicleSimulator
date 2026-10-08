#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hydrodynamic data of the MSS toolbox (MSS HYDRO folder): readers for WAMIT output and the power-based (constant
equivalent) matrices of computeManeuveringModel.m.

Each function names its MSS m-file (MSS: github.com/cybergalactic/MSS).
"""

import os
import re

import numpy as np
import math
from .numerics import pchip, trapz

#------------------------------------------------------------------------------

T6 = np.array([1, -1, -1, 1, -1, -1.0])                    # WAMIT axes (x fwd, y port, z up) -> Fossen axes
MIRROR = np.array([1, -1, 1, -1, 1, -1.0])                 # MSS counts the wave angle the other way round

#------------------------------------------------------------------------------

def _floats(pattern, text):
    return [float(s) for s in re.search(pattern, text).groups()]

#------------------------------------------------------------------------------

def read_wamit(prefix, Lpp=246.0, Boa=46.0, T=10.0):
    """Read the WAMIT v6 files prefix.1/.3/.4/.frc/.out (as MSS wamit2vessel.m) and return a dict with the main
    particulars (Lpp, Boa, T, rho, g, m, nabla, CG r_g, CB r_b) and
    MRB, C, Bext (6 x 6); freqs (w = 0, the WAMIT frequencies, w = inf); A, B (6, 6, len(freqs));
    w_rao, headings [deg]; forceRAO [N/m, N m/m] and motionRAO [m/m, rad/m], complex, shape (6, len(w_rao), 19).
    All about the CO (Lpp/2, centre line, waterline) in Fossen axes (x fwd, y stbd, z down). prefix is the path of
    the files without the extension, e.g. <MSS>/HYDRO/vessels_wamit/tanker/tanker; Lpp, Boa, T are the main
    dimensions (wamit2vessel.m takes them as inputs; the defaults are those of the MSS tanker)."""
    out = open(prefix + ".out").read()
    g = _floats(r"Gravity:\s+(\S+)", out)[0]
    L = _floats(r"Length scale:\s+(\S+)", out)[0]          # ULEN: all data are nondimensional with L
    rho = _floats(r"Water density:\s+(\S+)", out)[0]
    vol = _floats(r"VOLZ\):\s+(\S+)\s+(\S+)\s+(\S+)", out)
    cb = _floats(r"Center of Buoyancy \(Xb,Yb,Zb\):\s+(\S+)\s+(\S+)\s+(\S+)", out)
    cg = _floats(r"Center of Gravity\s+\(Xg,Yg,Zg\):\s+(\S+)\s+(\S+)\s+(\S+)", out)
    c3 = _floats(r"C\(3,3\),C\(3,4\),C\(3,5\):\s+(\S+)\s+(\S+)\s+(\S+)", out)
    c4 = _floats(r"C\(4,4\),C\(4,5\),C\(4,6\):\s+(\S+)\s+(\S+)\s+(\S+)", out)
    c5 = _floats(r"C\(5,5\),C\(5,6\):\s+(\S+)\s+(\S+)", out)
    rg = rho * g

    # hydrostatic and gravitational restoring coefficients: C33 * rho g L^2, C34, C35 * rho g L^3, others * rho g L^4
    Cw = np.zeros((6, 6))
    Cw[2, 2:5] = np.array(c3) * rg * np.array([L**2, L**3, L**3])
    Cw[3, 3:6] = np.array(c4) * rg * L**4
    Cw[4, 4:6] = np.array(c5) * rg * L**4
    Cw[3, 2], Cw[4, 2], Cw[4, 3] = Cw[2, 3], Cw[2, 4], Cw[3, 4]

    # tanker.frc (IALTFRC = 2), lines counted from 0: 2 rho, 3 CG, 5-10 mass matrix, 12-17 external damping
    frc = [line.split() for line in open(prefix + ".frc")]
    MRBw = np.array([[float(x) for x in frc[i]] for i in range(5, 11)])
    Bextw = np.array([[float(x) for x in frc[i]] for i in range(12, 18)])

    # tanker.1: PER I J Abar [Bbar];  PER = -1: w = 0,  PER = 0: w = inf;  A = Abar rho L^k,  B = Bbar rho w L^k
    rows = [line.split() for line in open(prefix + ".1") if line.strip()]
    per = np.array([float(r[0]) for r in rows])
    pers = np.unique(per)
    freqs = np.array([0.0 if p == -1 else np.inf if p == 0 else 2 * np.pi / p for p in pers])
    order = np.argsort(freqs)
    pers, freqs = pers[order], freqs[order]
    k = np.full((6, 6), 4); k[:3, :3] = 3; k[3:, 3:] = 5
    A = np.zeros((6, 6, len(freqs))); Bp = np.zeros((6, 6, len(freqs)))
    for r in rows:
        n = int(np.where(pers == float(r[0]))[0][0])
        i, j = int(r[1]) - 1, int(r[2]) - 1
        A[i, j, n] = float(r[3]) * rho * L**k[i, j]
        if len(r) > 4:                                     # no damping at w = 0 and w = inf
            Bp[i, j, n] = float(r[4]) * rho * freqs[n] * L**k[i, j]

    def rao_file(fname):                                   # PER BETA I Mod Pha Re Im
        d = np.loadtxt(fname)
        up, ub = np.unique(d[:, 0]), np.unique(d[:, 1])
        X = np.zeros((6, len(up), len(ub)), complex)
        X[d[:, 2].astype(int) - 1, np.searchsorted(up, d[:, 0]), np.searchsorted(ub, d[:, 1])] = d[:, 5] + 1j * d[:, 6]
        wr = 2 * np.pi / up
        o = np.argsort(wr)
        return wr[o], ub, X[:, o, :]

    w_rao, headings, X3 = rao_file(prefix + ".3")
    _, _, X4 = rao_file(prefix + ".4")
    sign = (T6 * MIRROR)[:, None, None]
    F = X3 * (rg * np.array([L**2] * 3 + [L**3] * 3))[:, None, None] * sign      # forces per metre wave amplitude
    R = X4 / np.array([1, 1, 1, L, L, L])[:, None, None] * sign                   # motions per metre wave amplitude

    TT = np.outer(T6, T6)
    return dict(name=os.path.basename(prefix), rho=rho, g=g, ulen=L, Lpp=Lpp, Boa=Boa, T=T,
                nabla=np.mean(vol), m=MRBw[0, 0], rg=T6[:3] * np.array(cg), rb=T6[:3] * np.array(cb),
                MRB=TT * MRBw, C=TT * Cw, Bext=TT * Bextw, freqs=freqs,
                A=A * TT[:, :, None], B=Bp * TT[:, :, None],
                w_rao=w_rao, headings=headings, forceRAO=F, motionRAO=R)


#------------------------------------------------------------------------------

def pm_weight(w, wp, b=0.74):
    """Normalized PM weighting of the notes and MSS: S ~ w^-5 exp(-b (wp/w)^4), unit area."""
    S = w**-5.0 * np.exp(-b * (wp / w)**4)
    return S / trapz(S, w)

#------------------------------------------------------------------------------

def power_based(v, wp, kappa_126=(0.05, 0.05, 0.05), dzeta_345=(0.0, 0.1, 0.0), n=100, b=0.74):
    """MSS computeManeuveringModel: power-based A_eq, B_eq; Bv = kappa B_eq (DOF 1, 2, 6), 2 dzeta sqrt(M G)
    (DOF 3, 4, 5); D = B_eq + Bv; T_126 = M_ii / D_ii (Lecture 5, Section 5.6)."""
    fin = np.isfinite(v["freqs"])
    w = v["freqs"][fin]
    wf = np.linspace(max(w.min(), 1e-6), w.max(), n)
    SN = pm_weight(wf, wp, b)
    Aeq = np.array([[trapz(pchip(w, v["A"][i, j, fin], wf) * SN, wf) for j in range(6)] for i in range(6)])
    Beq = np.array([[trapz(pchip(w, v["B"][i, j, fin], wf) * SN, wf) for j in range(6)] for i in range(6)])
    M = v["MRB"] + Aeq
    G = np.zeros((6, 6))
    G[2:5, 2:5] = v["C"][2:5, 2:5]
    Bv = np.zeros((6, 6))
    for kap, i in zip(kappa_126, (0, 1, 5)):
        Bv[i, i] = kap * Beq[i, i]
    for dz, i in zip(dzeta_345, (2, 3, 4)):
        Bv[i, i] = 2 * dz * np.sqrt(M[i, i] * G[i, i])
    D = Beq + Bv
    return dict(A_eq=Aeq, B_eq=Beq, Bv=Bv, D=D, M=M, T_126=np.array([M[i, i] / D[i, i] for i in (0, 1, 5)]))
