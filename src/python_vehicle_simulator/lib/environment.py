#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environmental models of the MSS toolbox (MSS LIBRARY/environment and LIBRARY/motionSickness): wave spectra,
encounter frequency, wind and current loads, motion sickness incidence.

Each function names its MSS m-file (MSS: github.com/cybergalactic/MSS).
"""

import numpy as np
import math
from .gnc import G_MSS
from .gnc import D2R
from .numerics import interp1

#------------------------------------------------------------------------------

def encounter(chi, U, Omega):
    """Encounter frequency w_e = w - w^2 U cos(chi) / g (MSS encounter.m; chi = 0 following sea, pi head sea)."""
    return Omega - (np.asarray(Omega)**2 * U * np.cos(chi) / G_MSS)

#------------------------------------------------------------------------------

def ISOmsi(t):
    """ISO 2631 MSI of 10 % after t hours (MSS ISOmsi.m): f = 0.1:0.03:0.63 Hz, a_z = 0.5 sqrt(2/t) for f < 0.315 Hz
    and 0.5 sqrt(2/t) 6.8837 f^1.67 for f >= 0.315 Hz. Returns (a_z in m/s^2, w_e = 2 pi f in rad/s)."""
    f = 0.1 + 0.03 * np.arange(18)
    t = max(t, np.finfo(float).eps)
    a_z = 0.5 * np.sqrt(2 / t) * np.ones_like(f)
    hi = f >= 0.315
    a_z[hi] = a_z[hi] * (6.8837 * f[hi]**1.67)
    return a_z, 2 * np.pi * f

#------------------------------------------------------------------------------

def _erf(x):
    return np.vectorize(math.erf)(x)

#------------------------------------------------------------------------------

def HMmsi(a_z, w_e):
    """O'Hanlon and McCauley (1974) MSI in percent after two hours (MSS HMmsi.m): a_z is replaced by its rms value
    over the vector, mu = -0.819 + 2.32 (log10 w_e)^2 with w_e in rad/s,
    MSI = 50 (1 - erf((-log10(a_z,rms/g) + mu)/(0.4 sqrt(2))))."""
    a_z = np.atleast_1d(np.asarray(a_z, float)).ravel()
    w_e = np.atleast_1d(np.asarray(w_e, float)).ravel()
    a_rms = np.sqrt(np.mean(a_z**2))
    mu = -0.819 + 2.32 * np.log10(np.maximum(w_e, 1e-6))**2
    I = (-np.log10(a_rms / 9.81) + mu) / (0.4 * np.sqrt(2))
    return 100 * (0.5 * (1 - _erf(I)))

#------------------------------------------------------------------------------

RHO_AIR = 1.224                     # blendermann94.m and isherwood72.m (comment "Density of air at 15 C")

#------------------------------------------------------------------------------

# Blendermann (1994): [CD_t, CD_l,AF(0), CD_l,AF(pi), delta, kappa] for the 17 ship types of blendermann94.m
BLENDERMANN = np.array([
    [0.95, 0.55, 0.60, 0.80, 1.2], [0.85, 0.65, 0.55, 0.40, 1.7], [0.85, 0.55, 0.50, 0.40, 1.4],
    [0.90, 0.55, 0.55, 0.40, 1.4], [0.85, 0.60, 0.65, 0.65, 1.1], [0.90, 0.60, 0.80, 0.55, 1.7],
    [1.00, 0.5 * (0.70 + 1.00), 0.5 * (0.75 + 1.10), 0.10, 1.7], [0.90, 0.45, 0.50, 0.80, 1.1],
    [0.95, 0.70, 0.70, 0.40, 1.1], [0.70, 0.60, 0.65, 0.50, 1.1], [0.90, 0.55, 0.80, 0.55, 1.2],
    [0.90, 0.40, 0.40, 0.80, 1.2], [0.85, 0.55, 0.65, 0.60, 1.4], [0.90, 0.55, 0.60, 0.60, 1.1],
    [0.70, 0.90, 0.55, 0.40, 3.1], [0.70, 0.75, 0.55, 0.40, 2.2], [0.85, 0.55, 0.55, 0.65, 1.1]])

#------------------------------------------------------------------------------

BLENDERMANN_TYPES = ["car carrier", "cargo vessel, loaded", "cargo vessel, container on deck",
                     "container ship, loaded", "destroyer", "diving support vessel", "drilling vessel", "ferry",
                     "fishing vessel", "liquefied natural gas tanker", "offshore supply vessel", "passenger liner",
                     "research vessel", "speed boat", "tanker, loaded", "tanker, in ballast", "tender"]

#------------------------------------------------------------------------------

def blendermann94(gamma_r, V_r, AFw, ALw, sH, sL, Loa, vessel_no):
    """Wind forces [X, Y, N] and coefficients CX, CY, CK, CN of Blendermann (1994) (MSS blendermann94.m).
    gamma_r: relative wind angle (rad), V_r: relative wind speed (m/s), AFw, ALw: frontal and lateral projected areas
    (m^2), Loa: length overall (m), vessel_no: 1-17. As in the m-file, CK uses sH/H_M with H_M = ALw/Loa and CN uses
    sL/Loa (sH: height of the centroid of ALw above the waterline, sL: its distance from the main section)."""
    gamma_r = np.mod(np.atleast_1d(np.asarray(gamma_r, float)) + np.pi, 2 * np.pi) - np.pi    # [-pi, pi)
    CDt, CDl_bow, CDl_stern, delta, kappa = BLENDERMANN[vessel_no - 1]
    Hm = ALw / Loa
    CDlAF = np.where(np.abs(gamma_r) <= np.pi / 2, CDl_bow, CDl_stern)
    CDl = CDlAF * AFw / ALw
    den = 1 - 0.5 * delta * (1 - CDl / CDt) * np.sin(2 * gamma_r)**2
    CX = -CDlAF * np.cos(gamma_r) / den
    CY = CDt * np.sin(gamma_r) / den
    CK = kappa * (sH / Hm) * CY
    CN = (sL / Loa - 0.18 * (np.abs(gamma_r) - np.pi / 2)) * CY        # symmetric in gamma_r
    q = 0.5 * RHO_AIR * V_r**2
    tau = np.array([q * CX * AFw, q * CY * ALw, q * CN * ALw * Loa])
    return tau, CX, CY, CK, CN

#------------------------------------------------------------------------------

# Isherwood (1972) regression tables of isherwood72.m: rows for 0, 10, ..., 180 deg
ISH_CX = np.array([
    [2.152, -5.00, 0.243, -0.164, 0, 0, 0], [1.714, -3.33, 0.145, -0.121, 0, 0, 0],
    [1.818, -3.97, 0.211, -0.143, 0, 0, 0.033], [1.965, -4.81, 0.243, -0.154, 0, 0, 0.041],
    [2.333, -5.99, 0.247, -0.190, 0, 0, 0.042], [1.726, -6.54, 0.189, -0.173, 0.348, 0, 0.048],
    [0.913, -4.68, 0, -0.104, 0.482, 0, 0.052], [0.457, -2.88, 0, -0.068, 0.346, 0, 0.043],
    [0.341, -0.91, 0, -0.031, 0, 0, 0.032], [0.355, 0, 0, 0, -0.247, 0, 0.018],
    [0.601, 0, 0, 0, -0.372, 0, -0.020], [0.651, 1.29, 0, 0, -0.582, 0, -0.031],
    [0.564, 2.54, 0, 0, -0.748, 0, -0.024], [-0.142, 3.58, 0, 0.047, -0.700, 0, -0.028],
    [-0.677, 3.64, 0, 0.069, -0.529, 0, -0.032], [-0.723, 3.14, 0, 0.064, -0.475, 0, -0.032],
    [-2.148, 2.56, 0, 0.081, 0, 1.27, -0.027], [-2.707, 3.97, -0.175, 0.126, 0, 1.81, 0],
    [-2.529, 3.76, -0.174, 0.128, 0, 1.55, 0]])

#------------------------------------------------------------------------------

ISH_CY = np.array([
    [0, 0, 0, 0, 0, 0, 0], [0.096, 0.22, 0, 0, 0, 0, 0], [0.176, 0.71, 0, 0, 0, 0, 0],
    [0.225, 1.38, 0, 0.023, 0, -0.29, 0], [0.329, 1.82, 0, 0.043, 0, -0.59, 0],
    [1.164, 1.26, 0.121, 0, -0.242, -0.95, 0], [1.163, 0.96, 0.101, 0, -0.177, -0.88, 0],
    [0.916, 0.53, 0.069, 0, 0, -0.65, 0], [0.844, 0.55, 0.082, 0, 0, -0.54, 0],
    [0.889, 0, 0.138, 0, 0, -0.66, 0], [0.799, 0, 0.155, 0, 0, -0.55, 0],
    [0.797, 0, 0.151, 0, 0, -0.55, 0], [0.996, 0, 0.184, 0, -0.212, -0.66, 0.34],
    [1.014, 0, 0.191, 0, -0.280, -0.69, 0.44], [0.784, 0, 0.166, 0, -0.209, -0.53, 0.38],
    [0.536, 0, 0.176, -0.029, -0.163, 0, 0.27], [0.251, 0, 0.106, -0.022, 0, 0, 0],
    [0.125, 0, 0.046, -0.012, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])

#------------------------------------------------------------------------------

ISH_CN = np.array([
    [0, 0, 0, 0, 0, 0], [0.0596, 0.061, 0, 0, 0, -0.074], [0.1106, 0.204, 0, 0, 0, -0.170],
    [0.2258, 0.245, 0, 0, 0, -0.380], [0.2017, 0.457, 0, 0.0067, 0, -0.472], [0.1759, 0.573, 0, 0.0118, 0, -0.523],
    [0.1925, 0.480, 0, 0.0115, 0, -0.546], [0.2133, 0.315, 0, 0.0081, 0, -0.526], [0.1827, 0.254, 0, 0.0053, 0, -0.443],
    [0.2627, 0, 0, 0, 0, -0.508], [0.2102, 0, -0.0195, 0, 0.0335, -0.492], [0.1567, 0, -0.0258, 0, 0.0497, -0.457],
    [0.0801, 0, -0.0311, 0, 0.0740, -0.396], [-0.0189, 0, -0.0488, 0.0101, 0.1128, -0.420],
    [0.0256, 0, -0.0422, 0.0100, 0.0889, -0.463], [0.0552, 0, -0.0381, 0.0109, 0.0689, -0.476],
    [0.0881, 0, -0.0306, 0.0091, 0.0366, -0.415], [0.0851, 0, -0.0122, 0.0025, 0, -0.220], [0, 0, 0, 0, 0, 0]])

#------------------------------------------------------------------------------

ISH_DEG = np.arange(0.0, 181.0, 10.0)

#------------------------------------------------------------------------------

def isherwood72(gamma_r, V_r, Loa, B, ALw, AFw, A_SS, S, C, M):
    """Wind forces [X, Y, N] and coefficients CX, CY, CN of Isherwood (1972) (MSS isherwood72.m; note the argument
    order ALw, AFw). gamma_r is wrapped to [-pi, pi): the tables use its magnitude (0-180 deg, element by element)
    and the side force and yaw moment its sign; CY and CN are returned for |gamma_r|."""
    gamma_r = np.mod(np.atleast_1d(np.asarray(gamma_r, float)) + np.pi, 2 * np.pi) - np.pi
    g_deg = np.abs(gamma_r / D2R)
    A, Bc, Cc = (interp1(ISH_DEG, T, g_deg) for T in (ISH_CX, ISH_CY, ISH_CN))
    p = np.array([1.0, 2 * ALw / Loa**2, 2 * AFw / B**2, Loa / B, S / Loa, C / Loa])
    CX = -(A[:, :6] @ p + A[:, 6] * M)
    CY = Bc[:, :6] @ p + Bc[:, 6] * A_SS / ALw
    CN = Cc @ p
    q = 0.5 * RHO_AIR * V_r**2
    sg = np.sign(gamma_r)
    tau = np.array([q * CX * AFw, q * CY * ALw * sg, q * CN * ALw * Loa * sg])
    return tau, CX, CY, CN

#------------------------------------------------------------------------------

def torsetSpectrum(Hs, wo, W, components=False):
    """Torsethaugen two-peaked spectrum S(W) [m^2 s] (MSS torsetSpectrum.m, G. Kleiven, Norsk Hydro).
    components=True (course addition) returns the wind-sea and swell parts (S_wind, S_swell); their sum is S."""
    W = np.array(W, float).ravel()
    if W[0] == 0:
        W[0] = 1e-4
    Tp = 2 * np.pi / wo
    if Hs <= 0:
        return (np.zeros_like(W), np.zeros_like(W)) if components else np.zeros_like(W)
    af, ae, au, a10, a1, kg, kg0, kg1, r = 6.6, 2.0, 25.0, 0.7, 0.5, 35.0, 3.5, 1.0, 0.857
    k0, k00, m0, b1, a20, a2, a3, s0, s1, b2, b3 = 0.5, 3.2, 4.0, 2.0, 0.6, 0.3, 6.0, 0.08, 3.0, 0.7, 3.0
    sigma_a2, sigma_b2 = 2 * 0.07**2, 2 * 0.09**2

    def g0(n, m):
        arg = (n - 1) / m
        return 1.0 / ((1.0 / m) * math.gamma(arg) * ((n / m)**(-arg))) if arg < 0 else \
            1.0 / ((1.0 / m) * math.gamma(arg) / ((n / m)**arg))

    def f12(n, m):
        a1m, b1m, c1m = 4.1, 2.0 * m**0.28 - 5.3, -1.45 * m**0.1 + 0.96
        a2m, b2m, c2m = 2.2 / m**3.3 + 0.57, -0.58 * m**0.37 + 0.53, -1.04 / m**1.9 + 0.94
        f1 = a1m / (n - b1m)**(-c1m) if c1m < 0 else a1m * (n - b1m)**c1m
        f2 = a2m / n**(-b2m) + c2m if b2m < 0 else a2m * n**b2m + c2m
        return f1, f2

    tf = af * Hs**(1.0 / 3.0)
    if Tp < tf:                                          # wind-dominated sea
        tl = ae * Hs**0.5
        eps1 = (tf - Tp) / (tf - tl)
        rpw = (1 - a10) * np.exp(-(eps1 / a1)**2) + a10
        hsw, hss = rpw * Hs, np.sqrt(1.0 - rpw**2) * Hs
        tpw, tps = Tp, tf + b1
        sp = (2 * np.pi / 9.81) * hsw / Tp**2
        gammaw, gammas = kg * (1 + kg0 * np.exp(-Hs / kg1)) * sp**r, 1.0
        nw = k0 * np.sqrt(Hs) + k00
        mw = m0
        ns, ms = nw, max(mw, 1.0)
    else:                                                # swell-dominated sea
        epsu = (Tp - tf) / (au - tf)
        rps = (1.0 - a20) * np.exp(-(epsu / a2)**2) + a20
        hss, hsw = rps * Hs, np.sqrt(1.0 - rps**2) * Hs
        tps = Tp
        ns, ms = k0 * np.sqrt(Hs) + k00, m0
        nw, mw = ns, m0 * (1 - b2 * np.exp(-Hs / b3))
        s4 = s0 * (1 - np.exp(-Hs / s1))
        tpw = ((g0(nw, mw) * hsw**2) / (16 * s4 * 0.4**nw))**(1.0 / (nw - 1.0))
        sf = (2 * np.pi / 9.81) * Hs / tf**2
        gammaw = 1.0
        gammas = kg * (1 + kg0 * np.exp(-Hs / kg1)) * sf**r * (1.0 + a3 * epsu)
    g0w, g0s = g0(nw, mw), g0(ns, ms)
    f1w, f2w = f12(nw, mw)
    f1s, f2s = f12(ns, ms)
    agammaw = (1 + f1w * np.log(gammaw)**f2w) / gammaw
    agammas = (1 + f1s * np.log(gammas)**f2s) / gammas
    f = W / (2 * np.pi)

    def peak(tp, hs, gam, ag, g0x, n, m):
        fn = f * tp
        sig = np.where(np.arange(len(fn)) <= np.nonzero(fn < 1)[0][-1], sigma_a2, sigma_b2)
        return g0x * ag * fn**(-n) * np.exp(-(n / m) * fn**(-m)) * gam**np.exp(-((fn - 1)**2) / sig) \
            * (tp * hs**2 / 16) / (2 * np.pi)
    S_wind, S_swell = peak(tpw, hsw, gammaw, agammaw, g0w, nw, mw), peak(tps, hss, gammas, agammas, g0s, ns, ms)
    return (S_wind, S_swell) if components else S_wind + S_swell

#------------------------------------------------------------------------------

def waveSpectrum(spectrumType, Parameter, W):
    """Wave spectrum S(W) [m^2 s] at the frequencies W [rad/s] (MSS waveSpectrum.m; W(1) = 0 is replaced by 1e-4):
      1 Bretschneider (A, B)        2 Pierson-Moskowitz (V_wind at 20 m)     3/4/5 ITTC-modified PM (Hs, T0 / T1 / Tz)
      6 JONSWAP (V_wind at 10 m, fetch)      7 JONSWAP (Hs, w0, gamma) with the DNV factor 1 - 0.287 ln(gamma)
      8 Torsethaugen (Hs, w0)."""
    W = np.array(W, float).ravel()
    if W[0] == 0:
        W[0] = 1e-4
    g = G_MSS
    p = Parameter
    PM = lambda A, B: A * W**(-5) * np.exp(-B / W**4)
    if spectrumType == 1:
        return PM(p[0], p[1])
    if spectrumType == 2:
        return PM(8.1e-3 * g**2, 0.74 * (g / p[0])**4)
    if spectrumType == 3:
        return PM(487 * p[0]**2 / p[1]**4, 1949 / p[1]**4)
    if spectrumType == 4:
        return PM(173 * p[0]**2 / p[1]**4, 691 / p[1]**4)
    if spectrumType == 5:
        return PM(123 * p[0]**2 / p[1]**4, 495 / p[1]**4)
    if spectrumType == 6:
        Vw10, fetch = p
        xt = g * fetch / Vw10**2
        w0 = 2 * np.pi * 3.5 * (g / Vw10) * xt**-0.33
        alpha = 0.076 * xt**(-0.22)
        sigma = np.where(W < w0, 0.07, 0.09)
        return alpha * g**2 * W**-5 * np.exp(-(5 / 4) * (w0 / W)**4) * 3.3**np.exp(-(W - w0)**2 / (2 * (sigma * w0)**2))
    if spectrumType == 7:
        Hs, w0, gamma = p
        alpha = (5 / 16) * Hs**2 * w0**4 / g**2
        if gamma < 1 or gamma > 7:                       # DNV formula for gamma
            k = 2 * np.pi / (w0 * np.sqrt(Hs))
            gamma = 5.0 if k <= 3.6 else (np.exp(5.75 - 1.15 * k) if k <= 5 else 1.0)
        sigma = np.where(W < w0, 0.07, 0.09)
        S1 = alpha * g**2 * W**-5 * np.exp(-(5 / 4) * (w0 / W)**4)
        S2 = gamma**np.exp(-(W - w0)**2 / (2 * (sigma * w0)**2))
        return (1 - 0.287 * np.log(gamma)) * S1 * S2
    if spectrumType == 8:
        return torsetSpectrum(p[0], p[1], W)
    raise ValueError("Wrong spectrum type identifier, spectrumType = 1, 2,..,8")

#------------------------------------------------------------------------------

def vw2hs(Vw):
    """Significant wave height Hs = 0.21 Vw^2 / g of a fully developed sea (MSS vw2hs.m)."""
    return 0.21 * Vw**2 / G_MSS

#------------------------------------------------------------------------------

def hs2vw(Hs):
    """Wind speed Vw = sqrt(Hs g / 0.21) (MSS hs2vw.m)."""
    return np.sqrt(Hs * G_MSS / 0.21)

#------------------------------------------------------------------------------

def waveDirectionalSpectrum(spectrumType, Parameters, numFreqIntervals, omegaMax, spreadingFlag=False,
                            numDirections=16, r_omega=None, r_mu=None, rng=None):
    """S_M(w, mu) = S(w) M(mu), random frequencies and directions, and the amplitudes A = sqrt(2 S_M dw dmu)
    (MSS waveDirectionalSpectrum.m). spectrumType: 'Modified PM' (Hs, w0), 'JONSWAP' (Hs, w0, gamma),
    'Torsethaugen' (Hs, w0). The uniform random numbers r_omega (one per frequency interval) and r_mu (one per
    direction) can be given, so that a run of the m-file can be repeated exactly. Returns S_M, Omega, Amp, S, M, mu.
    Not ported: the messages of the m-file and its JONSWAP check (error for w0 sqrt(Hs) > 1.75)."""
    rng = np.random.default_rng() if rng is None else rng
    Hs, w0 = Parameters[0], Parameters[1]
    if spreadingFlag:
        dmu = np.pi / (numDirections - 1)
        r_mu = rng.random(numDirections) if r_mu is None else np.asarray(r_mu, float)
        mu = -np.pi / 2 + np.arange(numDirections) * dmu + (r_mu - 0.5) * dmu
        mu = np.clip(mu, -np.pi / 2, np.pi / 2)
        M = (2 / np.pi) * np.cos(mu)**2
        trap = getattr(np, "trapezoid", None) or np.trapz
        M = M / trap(M, mu)
    else:
        mu, M, dmu = np.array([0.0]), np.array([1.0]), 1.0
    T0 = 2 * np.pi / w0
    Tz = 0.710 * T0
    dw = omegaMax / numFreqIntervals
    r_omega = rng.random(numFreqIntervals) if r_omega is None else np.asarray(r_omega, float)
    Omega = np.arange(numFreqIntervals) * dw + dw / 2 + (r_omega - 0.5) * dw
    if spectrumType == "Modified PM":
        S = waveSpectrum(5, [Hs, Tz], Omega)
    elif spectrumType == "JONSWAP":
        S = waveSpectrum(7, [Hs, w0, Parameters[2]], Omega)
    else:
        S = waveSpectrum(8, [Hs, w0], Omega)
    S_M = np.outer(S, M)
    return S_M, Omega, np.sqrt(2 * S_M * dw * dmu), S, M, mu

#------------------------------------------------------------------------------

class ForceRAO:
    """1st-order wave forces from force RAOs (MSS waveForceRAO.m) or wave-frequency motions from motion RAOs (MSS
    waveMotionRAO.m, kind='motion'), with the wave model of MSS waveInitialization.m: the Re and Im parts of the
    zero-speed RAO tables are interpolated linearly (with extrapolation) in frequency at the wave frequencies Omega;
    the direction table is closed at 2 pi with the column of 0 deg, and the RAOs are interpolated linearly in direction
    at beta_wave - psi + mu wrapped to [0, 2 pi). phases: the random phase angles (numFreq x numDirections), the field
    randomPhases of the waveModel of waveInitialization.m (drawn there with rand); here they are an argument."""

    def __init__(self, data, Omega, mu, phases, kind="force"):
        self.w, self.headings, self.g = data["w"], data["headings"], data["main"]["g"]
        amp, ph = data[kind + "_amp"], data[kind + "_phase"]
        self.Omega, self.mu = np.asarray(Omega, float), np.atleast_1d(np.asarray(mu, float))
        self.phases = np.asarray(phases, float).reshape(len(self.Omega), len(self.mu))
        re, im = amp * np.cos(ph), amp * np.sin(ph)                     # 6 x nw x nh
        self.re = np.stack([interp1(self.w, re[i], self.Omega, extrap=True) for i in range(6)])   # 6 x nO x nh
        self.im = np.stack([interp1(self.w, im[i], self.Omega, extrap=True) for i in range(6)])
        self.raoAngles = np.r_[self.headings, 2 * np.pi]                 # the table closed at 2 pi (0 deg column)
        self.re = np.concatenate([self.re, self.re[:, :, :1]], axis=2)
        self.im = np.concatenate([self.im, self.im[:, :, :1]], axis=2)
        self.kind = kind

    def rao(self, beta_RAO):
        """Complex RAOs (6 x nO x nDir) at the relative directions beta_RAO (wrapped to [0, 2 pi))."""
        out = np.zeros((6, len(self.Omega), len(beta_RAO)), complex)
        for k, b in enumerate(beta_RAO):
            out[:, :, k] = (np.stack([interp1(self.raoAngles, self.re[i].T, b)[0] for i in range(6)])
                            + 1j * np.stack([interp1(self.raoAngles, self.im[i].T, b)[0] for i in range(6)]))
        return out

    def __call__(self, t, Amp, U, psi, beta_wave):
        """Returns the 6 forces (or the motions eta_w, nu_w, nudot_w for kind='motion') and the wave elevation."""
        Amp = np.asarray(Amp, float).reshape(len(self.Omega), len(self.mu))
        beta_RAO = np.mod(beta_wave - psi + self.mu, 2 * np.pi)
        Oe = np.abs(self.Omega[:, None] - (self.Omega[:, None]**2 / self.g) * U * np.cos(beta_RAO[None, :]))
        arg = Oe * t + self.phases
        elev = float(np.sum(Amp * np.cos(arg)))
        R = self.rao(beta_RAO)
        if self.kind == "force":
            return np.sum(np.abs(R) * Amp * np.cos(arg + np.angle(R)), axis=(1, 2)), elev
        V, A = 1j * Oe * R, -Oe**2 * R
        f = lambda X: np.sum(np.abs(X) * Amp * np.cos(arg + np.angle(X)), axis=(1, 2))
        return f(R), f(V), f(A), elev
