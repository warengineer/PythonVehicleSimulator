#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hydrodynamic data of the MSS toolbox (MSS HYDRO folder): readers for WAMIT output and the power-based (constant
equivalent) matrices of computeManeuveringModel.m.

Each function names its MSS m-file (MSS: github.com/cybergalactic/MSS).
"""

import re

import numpy as np
import math
from .numerics import pchip, trapz

#------------------------------------------------------------------------------

T6 = np.array([1, -1, -1, 1, -1, -1.0])                    # WAMIT axes (x fwd, y port, z up) -> Fossen axes

#------------------------------------------------------------------------------

def _floats(pattern, text):
    return [float(s) for s in re.search(pattern, text).groups()]

#------------------------------------------------------------------------------

def read_wamit_hydro(prefix, Lpp=246.0, Boa=46.0, T=10.0):
    """Read the WAMIT v6 files prefix.1, .frc and .out (as MSS wamit2vessel.m) and return a dict: rho, g, Lpp, Boa,
    T, nabla, m, r_g, MRB, C (restoring), freqs (w = 0, the WAMIT frequencies, w = inf) and A, B (6 x 6 x len(freqs))
    about the CO (Lpp/2, centre line, waterline), Fossen axes (x fwd, y stbd, z down). prefix is the path of the files
    without the extension, e.g. <MSS>/HYDRO/vessels_wamit/tanker/tanker; Lpp, Boa, T are the main dimensions of that
    tanker (wamit2vessel.m takes them as inputs)."""
    out = open(prefix + ".out").read()
    g = _floats(r"Gravity:\s+(\S+)", out)[0]
    L = _floats(r"Length scale:\s+(\S+)", out)[0]
    rho = _floats(r"Water density:\s+(\S+)", out)[0]
    vol = _floats(r"VOLZ\):\s+(\S+)\s+(\S+)\s+(\S+)", out)
    cg = _floats(r"Center of Gravity\s+\(Xg,Yg,Zg\):\s+(\S+)\s+(\S+)\s+(\S+)", out)
    c3 = _floats(r"C\(3,3\),C\(3,4\),C\(3,5\):\s+(\S+)\s+(\S+)\s+(\S+)", out)
    c4 = _floats(r"C\(4,4\),C\(4,5\),C\(4,6\):\s+(\S+)\s+(\S+)\s+(\S+)", out)
    c5 = _floats(r"C\(5,5\),C\(5,6\):\s+(\S+)\s+(\S+)", out)
    Cw = np.zeros((6, 6))
    Cw[2, 2:5] = np.array(c3) * rho * g * np.array([L**2, L**3, L**3])
    Cw[3, 3:6] = np.array(c4) * rho * g * L**4
    Cw[4, 4:6] = np.array(c5) * rho * g * L**4
    Cw[3, 2], Cw[4, 2], Cw[4, 3] = Cw[2, 3], Cw[2, 4], Cw[3, 4]
    frc = [line.split() for line in open(prefix + ".frc")]
    MRBw = np.array([[float(s) for s in frc[i]] for i in range(5, 11)])
    rows = [line.split() for line in open(prefix + ".1") if line.strip()]
    pers = np.unique([float(r[0]) for r in rows])
    freqs = np.array([0.0 if p == -1 else np.inf if p == 0 else 2 * np.pi / p for p in pers])
    order = np.argsort(freqs)
    pers, freqs = pers[order], freqs[order]
    k = np.full((6, 6), 4)
    k[:3, :3], k[3:, 3:] = 3, 5
    A = np.zeros((6, 6, len(freqs)))
    B = np.zeros((6, 6, len(freqs)))
    for r in rows:
        j = int(np.where(pers == float(r[0]))[0][0])
        a, b = int(r[1]) - 1, int(r[2]) - 1
        A[a, b, j] = float(r[3]) * rho * L**k[a, b]
        if len(r) > 4:
            B[a, b, j] = float(r[4]) * rho * freqs[j] * L**k[a, b]
    TT = np.outer(T6, T6)
    return dict(rho=rho, g=g, Lpp=Lpp, Boa=Boa, T=T, nabla=np.mean(vol), m=MRBw[0, 0], rg=T6[:3] * np.array(cg),
                MRB=TT * MRBw, C=TT * Cw, freqs=freqs, A=A * TT[:, :, None], B=B * TT[:, :, None])

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
