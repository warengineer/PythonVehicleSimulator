#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inertial navigation functions of the MSS toolbox (MSS INS folder): gravity model, attitude observers, aided INS.

Each function names its MSS m-file (MSS: github.com/cybergalactic/MSS).
"""

import numpy as np
import math
from .gnc import D2R, Rquat, Rzyx, Smtrx, Tzyx, expq, quatprod, ssa
from .numerics import expm, expm_taylor, invQR, rk4, vnorm

#------------------------------------------------------------------------------

def gravity(mu):
    """Acceleration of gravity (m/s^2) at latitude mu (rad), WGS-84 (MSS gravity.m)."""
    s2 = np.sin(mu) ** 2
    return 9.7803253359 * (1 + 0.001931850400 * s2) / np.sqrt(1 - 0.006694384442 * s2)

#------------------------------------------------------------------------------

I3 = np.eye(3)

#------------------------------------------------------------------------------

O3 = np.zeros((3, 3))

#------------------------------------------------------------------------------

# [city, North (nT), East (nT), Down (nT), longitude (deg), latitude (deg)] of magneticField.m (hypothetical
# demonstration values, as the m-file says; real values come from NOAA's geomagnetic calculator)
MAGNETIC_FIELD = [
    ("Trondheim", 13559, 921, 50209, 10.3951, 63.4305), ("Oslo", 13500, 900, 50000, 10.7522, 59.9139),
    ("New York", 22000, -5000, 45000, -74.0060, 40.7128), ("Tokyo", 30000, 2000, 35000, 139.6917, 35.6895),
    ("Sydney", 25000, 12000, -60000, 151.2093, -33.8688), ("Cape Town", 15000, 6000, -40000, 18.4241, -33.9249),
    ("Moscow", 12000, 500, 55000, 37.6173, 55.7558), ("Sao Paulo", 25000, -10000, -20000, -46.6333, -23.5505),
    ("Cairo", 35000, 1000, 30000, 31.2357, 30.0444), ("Beijing", 40000, 7000, 50000, 116.4074, 39.9042),
    ("London", 20000, -500, 48000, -0.1278, 51.5074), ("Paris", 21000, 600, 47000, 2.3522, 48.8566),
    ("Berlin", 19000, 400, 49000, 13.4050, 52.5200), ("Rome", 24000, 800, 46000, 12.4964, 41.9028),
    ("Madrid", 26000, -1000, 44000, -3.7038, 40.4168), ("Toronto", 20000, -4000, 43000, -79.3832, 43.6532),
    ("Los Angeles", 22000, 2000, 42000, -118.2437, 34.0522), ("Mexico City", 28000, 3000, 41000, -99.1332, 19.4326),
    ("Buenos Aires", 24000, -2000, -25000, -58.3816, -34.6037),
    ("Rio de Janeiro", 26000, -3000, -23000, -43.1729, -22.9068),
    ("Mumbai", 34000, 4000, 25000, 72.8777, 19.0760), ("Delhi", 33000, 3500, 26000, 77.1025, 28.7041),
    ("Shanghai", 38000, 5000, 30000, 121.4737, 31.2304), ("Hong Kong", 36000, 5500, 28000, 114.1694, 22.3193),
    ("Singapore", 32000, 4000, 10000, 103.8198, 1.3521), ("Bangkok", 31000, 3000, 29000, 100.5018, 13.7563),
    ("Istanbul", 23000, 700, 46000, 28.9784, 41.0082), ("Johannesburg", 16000, 6500, -42000, 28.0473, -26.2041),
    ("Nairobi", 18000, 5000, -30000, 36.8219, -1.2921), ("Moscow", 12000, 500, 55000, 37.6173, 55.7558)]

#------------------------------------------------------------------------------

def magneticField(index):
    """NED magnetic-field vector m_ref [nT], longitude l [rad], latitude mu [rad] and city name for index 1-30
    (MSS magneticField.m; hypothetical demonstration values)."""
    city, N, E, D, lon, lat = MAGNETIC_FIELD[index - 1]
    return np.array([N, E, D], float), lon * D2R, lat * D2R, city

#------------------------------------------------------------------------------

def ins_signal_accel(t, signalNo=1):
    """The BODY acceleration a(t) of insSignal.m (its specific force is a - R'g)."""
    if signalNo == 1:
        return np.array([0.1 * math.sin(0.1 * t), 0.01 * math.cos(0.1 * t), 0.05 * math.sin(0.05 * t)])
    return np.array([0.2 * math.sin(0.07 * t + 0.5) + 0.05 * math.sin(0.2 * t),
                     0.1 * math.cos(0.1 * t) + 0.02 * math.cos(0.3 * t + 0.3),
                     0.15 * math.sin(0.05 * t + 1.0) + 0.03 * math.cos(0.2 * t)])

#------------------------------------------------------------------------------

def ins_signal_rate(t, signalNo=1):
    """The BODY angular velocity w(t) of insSignal.m."""
    if signalNo == 1:
        return np.array([0.03 * math.cos(0.2 * t), -0.02 * math.sin(0.1 * t), 0.01 * math.sin(0.05 * t)])
    return np.array([0.05 * math.sin(0.3 * t) + 0.01 * math.cos(0.1 * t + 0.7),
                     0.04 * math.cos(0.2 * t + 1.2),
                     0.03 * math.sin(0.15 * t + 0.4)])

#------------------------------------------------------------------------------

def insSignal(x, h, t_k, mu, m_ref, signalNo=1, noise=None, accel=True):
    """One step of the MSS INS test-signal generator insSignal.m. x = [p(3) v(3) b_acc(3) Theta(3) b_ars(3)]
    (NED position and velocity, accelerometer bias, Euler angles, ARS bias). noise() returns the next standard
    normal number in the order of the m-file's randn calls (Gauss-Markov biases for signalNo = 2, then the
    specific-force, angular-rate and magnetometer noise, 3 numbers each). Returns x[k+1], f_imu, w_imu, m_imu.
    As in the m-file, rk4 holds the time t_k in all four stages. accel=False (not in MSS)
    sets the BODY acceleration to zero, so that the accelerometers measure gravity only."""
    T_acc = T_ars = 300.0
    sigma_b_acc = 20e-6 * 9.81
    sigma_b_ars = (8.0 / 3600.0) * D2R
    g_n = np.array([0.0, 0.0, gravity(mu)])
    if signalNo not in (1, 2):
        raise ValueError("signalNo must be 1 or 2.")

    def a_true(t):
        return ins_signal_accel(t, signalNo) if accel else np.zeros(3)

    def w_true(t):
        return ins_signal_rate(t, signalNo)

    def f_true(t, z, R=None):
        if R is None:
            R = Rzyx(z[9], z[10], z[11])
        return a_true(t) - R.T @ g_n

    a_k, w_k = a_true(t_k), w_true(t_k)            # rk4 holds t_k in all four stages

    def dynamics(z, t):
        R = Rzyx(z[9], z[10], z[11])
        zd = np.zeros(15)
        zd[0:3] = z[3:6]
        zd[3:6] = R @ (a_k - R.T @ g_n) + g_n
        zd[9:12] = Tzyx(z[9], z[10]) @ w_k
        return zd

    x = rk4(dynamics, h, np.asarray(x, float), t_k)
    if signalNo == 2:
        a_acc = math.exp(-h / T_acc)
        a_ars = math.exp(-h / T_ars)
        x[6:9] = a_acc * x[6:9] + sigma_b_acc * math.sqrt(1 - a_acc**2) * np.array([noise() for _ in range(3)])
        x[12:15] = a_ars * x[12:15] + sigma_b_ars * math.sqrt(1 - a_ars**2) * np.array([noise() for _ in range(3)])
    w1 = 0.005 * np.array([noise() for _ in range(3)])
    w2 = 0.005 * np.array([noise() for _ in range(3)])
    f_imu = f_true(t_k + h, x) + x[6:9] + w1
    w_imu = w_true(t_k + h) + x[12:15] + w2
    w3 = 0.005 * np.array([noise() for _ in range(3)])
    m_imu = Rzyx(x[9], x[10], x[11]).T @ m_ref + w3
    return x, f_imu, w_imu, m_imu

#------------------------------------------------------------------------------

def acc2rollpitch(f_imu, b_acc=None):
    """Static roll and pitch angles from specific force (MSS acc2rollpitch.m): phi = atan2(-f_y, -f_z),
    theta = atan2(f_x, sqrt(f_y^2 + f_z^2)). f_imu is (3,) or (n, 3); returns arrays of length n."""
    F = np.atleast_2d(np.asarray(f_imu, float))
    if F.shape == (3, 1):
        F = F.T
    if F.shape[1] != 3:
        raise ValueError("Input must have 3 columns representing [fx, fy, fz]")
    b = np.zeros(3) if b_acc is None else np.ravel(b_acc)
    f = F - b
    phi = np.arctan2(-f[:, 1], -f[:, 2])
    theta = np.arctan2(f[:, 0], np.sqrt(f[:, 1]**2 + f[:, 2]**2))
    return phi, theta

#------------------------------------------------------------------------------

def staticRollPitchYaw(f_imu, m_imu=None):
    """Static roll, pitch and magnetic heading (MSS staticRollPitchYaw.m): roll and pitch from acc2rollpitch, and the
    tilt-compensated magnetometer h = R_y(theta) R_x(phi) m with psi = atan2(-h_y, h_x) for FSD BODY axes.
    Returns (phi, theta) or (phi, theta, psi) as arrays (the input checks of the m-file are left out)."""
    F = np.atleast_2d(np.asarray(f_imu, float))
    phi, theta = acc2rollpitch(F)
    if m_imu is None:
        return phi, theta
    M = np.atleast_2d(np.asarray(m_imu, float))
    hx = M[:, 0] * np.cos(theta) + M[:, 1] * np.sin(phi) * np.sin(theta) + M[:, 2] * np.cos(phi) * np.sin(theta)
    hy = M[:, 1] * np.cos(phi) - M[:, 2] * np.sin(phi)
    return phi, theta, np.arctan2(-hy, hx)

#------------------------------------------------------------------------------

V01 = np.array([0.0, 0.0, -1.0])                 # normalized gravity reference (an IMU at rest measures -g)

#------------------------------------------------------------------------------

def quatObserver(quat, b_ars, h, Ki, k1, k2, m_ref, imu_meas):
    """Nonlinear quaternion attitude observer with ARS bias estimation (MSS quatObserver.m; Mahony et al. 2008 in
    the cross-product form of Grip et al. 2013). imu_meas = [f, w] (6), [f, w, psi] (7) or [f, w, m] (9).
    sigma = k1 v1 x R'v01 + k2 v2 x R'v02, q[k+1] = exp(Tquat(w - b + sigma) h) q[k], b[k+1] = b[k] - h Ki sigma."""
    imu_meas = np.ravel(imu_meas)
    Rt = Rquat(quat).T
    f_imu, w_imu = imu_meas[0:3], imu_meas[3:6]
    if imu_meas.size == 9:
        m_imu = imu_meas[6:9]
        v1 = f_imu / vnorm(f_imu)
        v02 = m_ref / vnorm(m_ref)
        v2 = m_imu / vnorm(m_imu)
        sigma = k1 * np.cross(v1, Rt @ V01) + k2 * np.cross(v2, Rt @ v02)
    elif imu_meas.size == 7:
        psi = imu_meas[6]
        v1 = f_imu / vnorm(f_imu)
        v02 = np.array([math.cos(psi), math.sin(psi), 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        sigma = k1 * np.cross(v1, Rt @ V01) + k2 * np.cross(v2, Rt @ v02)
    else:
        sigma = np.zeros(3)
    w_est = w_imu - b_ars + sigma
    quat = expq(w_est, h) @ quat
    quat = quat / vnorm(quat)
    b_ars = b_ars - h * (Ki @ sigma)
    return quat, b_ars

#------------------------------------------------------------------------------

def quatMEKF(quat, b_ars, P_prd, h, Qd, Rd, T_ars, m_ref, imu_meas):
    """Multiplicative EKF for quaternion attitude and ARS bias (MSS quatMEKF.m). Error state
    delta_x = [delta_a (2 x Gibbs vector), delta_b_ars]. The corrector runs when imu_meas has 7 (compass) or 9
    (magnetometer) elements; the predictor at every call. Returns quat, b_ars, P_prd."""
    imu_meas = np.ravel(imu_meas)
    R = Rquat(quat)
    Rt = R.T
    w_imu = imu_meas[3:6]
    if imu_meas.size == 6:
        P_hat = P_prd
    else:
        f_imu = imu_meas[0:3]
        v1 = f_imu / vnorm(f_imu)
        if imu_meas.size == 9:
            m_imu = imu_meas[6:9]
            v2 = m_imu / vnorm(m_imu)
            v02 = m_ref / vnorm(m_ref)
            Cd = np.zeros((6, 6))
            Cd[0:3, 0:3] = Smtrx(Rt @ V01)
            Cd[3:6, 0:3] = Smtrx(Rt @ v02)
            dy = np.concatenate((v1 - Rt @ V01, v2 - Rt @ v02))
        else:
            psi = imu_meas[6]
            c_psi = np.array([0.0, R[2, 1], R[2, 2]]) / (R[2, 1]**2 + R[2, 2]**2)
            Cd = np.vstack((np.hstack((Smtrx(Rt @ V01), O3)), np.concatenate((c_psi, np.zeros(3)))))
            dy = np.concatenate((v1 - Rt @ V01, [ssa(psi - math.atan2(R[1, 0], R[0, 0]))]))
        K = P_prd @ Cd.T @ invQR(Cd @ P_prd @ Cd.T + Rd)
        IKC = np.eye(P_prd.shape[0]) - K @ Cd
        dx = K @ dy
        P_hat = IKC @ P_prd @ IKC.T + K @ Rd @ K.T
        da = dx[0:3]
        dq = np.concatenate(([2.0], da)) / math.sqrt(4.0 + da @ da)
        b_ars = b_ars + dx[3:6]
        quat = quatprod(quat, dq)
        quat = quat / vnorm(quat)
    A = np.zeros((6, 6))
    A[0:3, 0:3] = -Smtrx(w_imu - b_ars)
    A[0:3, 3:6] = -I3
    A[3:6, 3:6] = -(1.0 / T_ars) * I3
    Ad = expm(A * h)
    Ed = h * np.diag([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
    P_prd = Ad @ P_hat @ Ad.T + Ed @ Qd @ Ed.T
    quat = expq(w_imu - b_ars, h) @ quat
    quat = quat / vnorm(quat)
    return quat, b_ars, P_prd

#------------------------------------------------------------------------------

def ins_ahrs(x_ins, P_prd, mu, h, Qd, Rd, T_acc, f_imu, y_ahrs, y_pos=None, y_vel=None):
    """Feedback ESKF for an INS whose attitude comes from an external AHRS (MSS ins_ahrs.m). x_ins = [p, v, b_acc]
    (9 states, 9 error states); corrector with y_pos (and y_vel) when given, predictor and INS propagation always."""
    p, v, b = x_ins[0:3].copy(), x_ins[3:6].copy(), x_ins[6:9].copy()
    g_n = np.array([0.0, 0.0, gravity(mu)])
    R = Rzyx(y_ahrs[0], y_ahrs[1], y_ahrs[2])
    f_ins = f_imu - b
    A = np.block([[O3, I3, O3], [O3, O3, -R], [O3, O3, -(1.0 / T_acc) * I3]])
    Ad = expm(A * h)
    Ed = h * np.block([[O3, O3], [-R, O3], [O3, I3]])
    if y_pos is None:
        P_hat = P_prd
    else:
        if y_vel is None:
            Cd = np.hstack((I3, O3, O3))
            eps = y_pos - p
        else:
            Cd = np.block([[I3, O3, O3], [O3, I3, O3]])
            eps = np.concatenate((y_pos - p, y_vel - v))
        K = P_prd @ Cd.T @ np.linalg.inv(Cd @ P_prd @ Cd.T + Rd)
        IKC = np.eye(9) - K @ Cd
        dx = K @ eps
        P_hat = IKC @ P_prd @ IKC.T + K @ Rd @ K.T
        p, v, b = p + dx[0:3], v + dx[3:6], b + dx[6:9]
    P_prd = Ad @ P_hat @ Ad.T + Ed @ Qd @ Ed.T
    a_ins = R @ f_ins + g_n                      # f_ins uses the bias before the reset, as in the m-file
    p = p + h * v + 0.5 * h**2 * a_ins
    v = v + h * a_ins
    return np.concatenate((p, v, b)), P_prd

#------------------------------------------------------------------------------

def euler_rate_jacobian(theta, w):
    """d(Tzyx(theta) w)/d theta: the local function eulerRateJacobian of MSS ins_euler.m."""
    phi, th = theta[0], theta[1]
    q, r = w[1], w[2]
    sphi, cphi, tth, cth = math.sin(phi), math.cos(phi), math.tan(th), math.cos(th)
    secth = 1.0 / cth
    d_dphi = [tth * (cphi * q - sphi * r), -sphi * q - cphi * r, secth * (cphi * q - sphi * r)]
    d_dtheta = [secth**2 * (sphi * q + cphi * r), 0.0, secth * tth * (sphi * q + cphi * r)]
    return np.column_stack((d_dphi, d_dtheta, np.zeros(3)))

#------------------------------------------------------------------------------

def ins_euler(x_ins, P_prd, mu, h, Qd, Rd, T_acc, T_ars, imu_meas, y_psi, y_pos=None, y_vel=None):
    """Feedback ESKF for an INS aided by position, a gravity reference vector and a compass, Euler angles (MSS
    ins_euler.m). x_ins = [p, v, b_acc, Theta, b_ars] (15 states, 15 additive error states). The corrector runs
    when y_pos is given; Ad = expm_taylor(A h) and the attitude is propagated with RK4, as in the m-file."""
    p, v, b_acc = x_ins[0:3].copy(), x_ins[3:6].copy(), x_ins[6:9].copy()
    th, b_ars = x_ins[9:12].copy(), x_ins[12:15].copy()
    g_n = np.array([0.0, 0.0, gravity(mu)])
    R = Rzyx(th[0], th[1], th[2])
    T = Tzyx(th[0], th[1])
    imu_meas = np.ravel(imu_meas)
    f_imu, w_imu = imu_meas[0:3], imu_meas[3:6]
    f_ins = f_imu - b_acc
    w_ins = w_imu - b_ars
    v1 = f_ins / vnorm(f_ins)
    if y_pos is not None:
        Sg = Smtrx(R.T @ V01) @ np.linalg.inv(T)            # Smtrx(R'*v01)/T
        row_psi = np.zeros(15)
        row_psi[11] = 1.0
        rows = [np.hstack((I3, O3, O3, O3, O3))]
        eps = [y_pos - p]
        if y_vel is not None:
            rows.append(np.hstack((O3, I3, O3, O3, O3)))
            eps.append(y_vel - v)
        rows += [np.hstack((O3, O3, O3, Sg, O3)), row_psi[None, :]]
        eps += [v1 - R.T @ V01, [ssa(y_psi - th[2])]]
        Cd = np.vstack(rows)
        eps = np.concatenate(eps)
        K = P_prd @ Cd.T @ np.linalg.inv(Cd @ P_prd @ Cd.T + Rd)
        IKC = np.eye(15) - K @ Cd
        dx = K @ eps
        P_hat = IKC @ P_prd @ IKC.T + K @ Rd @ K.T
        p, v, b_acc = p + dx[0:3], v + dx[3:6], b_acc + dx[6:9]
        th, b_ars = th + dx[9:12], b_ars + dx[12:15]
        R = Rzyx(th[0], th[1], th[2])
        T = Tzyx(th[0], th[1])
        f_ins = f_imu - b_acc
        w_ins = w_imu - b_ars
    else:
        P_hat = P_prd
    Ath = euler_rate_jacobian(th, w_ins)
    Ti = np.linalg.inv(T)
    A = np.zeros((15, 15))
    A[0:3, 3:6] = I3
    A[3:6, 6:9] = -R
    A[3:6, 9:12] = -R @ Smtrx(f_ins) @ Ti
    A[6:9, 6:9] = -(1.0 / T_acc) * I3
    A[9:12, 9:12] = Ath
    A[9:12, 12:15] = -T
    A[12:15, 12:15] = -(1.0 / T_ars) * I3
    Ad = expm_taylor(A * h)
    Ed = np.zeros((15, 12))
    Ed[3:6, 0:3] = -R
    Ed[6:9, 3:6] = I3
    Ed[9:12, 6:9] = -T
    Ed[12:15, 9:12] = I3
    Ed = h * Ed
    P_prd = Ad @ P_hat @ Ad.T + Ed @ Qd @ Ed.T
    a_ins = R @ f_ins + g_n
    p = p + h * v + 0.5 * h**2 * a_ins
    v = v + h * a_ins
    th = rk4(lambda z, w: Tzyx(z[0], z[1]) @ w, h, th, w_ins)
    return np.concatenate((p, v, b_acc, th, b_ars)), P_prd

#------------------------------------------------------------------------------

def _mekf_tail(p, v, b_acc, q, b_ars, P_hat, f_imu, w_imu, mu, h, Qd, T_acc, T_ars, pseudo, zint):
    """Shared predictor and INS propagation of ins_mekf.m and ins_mekf_psi.m."""
    g_n = np.array([0.0, 0.0, gravity(mu)])
    f_ins = f_imu - b_acc
    w_ins = w_imu - b_ars
    Rq = Rquat(q)
    A = np.zeros((15, 15))
    A[0:3, 3:6] = I3
    A[3:6, 6:9] = -Rq
    A[3:6, 9:12] = -Rq @ Smtrx(f_ins)
    A[6:9, 6:9] = -(1.0 / T_acc) * I3
    A[9:12, 9:12] = -Smtrx(w_ins)
    A[9:12, 12:15] = -I3
    A[12:15, 12:15] = -(1.0 / T_ars) * I3
    Ed = np.zeros((15, 12))
    Ed[3:6, 0:3] = -Rq
    Ed[6:9, 3:6] = I3
    Ed[9:12, 6:9] = -I3
    Ed[12:15, 9:12] = I3
    Ed = h * Ed
    if pseudo:
        A = np.block([[A, np.zeros((15, 1))], [np.zeros((1, 16))]])
        Ed = np.block([[Ed, np.zeros((15, 1))], [np.zeros((1, 12)), np.ones((1, 1))]])
    Ad = expm(A * h)
    P_prd = Ad @ P_hat @ Ad.T + Ed @ Qd @ Ed.T
    a_ins = Rq @ f_ins + g_n
    p = p + h * v + 0.5 * h**2 * a_ins
    v = v + h * a_ins
    q = expq(w_ins, h) @ q
    q = q / vnorm(q)
    x = np.concatenate((p, v, b_acc, q, b_ars))
    if pseudo:
        x = np.concatenate((x, [zint]))
    return x, P_prd

#------------------------------------------------------------------------------

def _mekf_reset(x_ins, P_prd, Cd, dy, Rmat, pseudo):
    """Shared corrector and reset of ins_mekf.m and ins_mekf_psi.m."""
    p, v, b_acc = x_ins[0:3].copy(), x_ins[3:6].copy(), x_ins[6:9].copy()
    q, b_ars = x_ins[9:13].copy(), x_ins[13:16].copy()
    zint = x_ins[16] if pseudo else 0.0
    if dy is None:
        return p, v, b_acc, q, b_ars, zint, P_prd
    K = P_prd @ Cd.T @ np.linalg.inv(Cd @ P_prd @ Cd.T + Rmat)
    IKC = np.eye(P_prd.shape[0]) - K @ Cd
    dx = K @ dy
    P_hat = IKC @ P_prd @ IKC.T + K @ Rmat @ K.T
    p, v = p + dx[0:3], v + dx[3:6]                       # reset of the INS states
    b_acc, b_ars = b_acc + dx[6:9], b_ars + dx[12:15]
    da = dx[9:12]
    dq = np.concatenate(([2.0], da)) / math.sqrt(4.0 + da @ da)
    q = quatprod(q, dq)
    q = q / vnorm(q)
    if pseudo:
        zint = zint + dx[15]
    return p, v, b_acc, q, b_ars, zint, P_hat

#------------------------------------------------------------------------------

def _blkdiag(blocks):
    n = sum(np.atleast_2d(b).shape[0] for b in blocks)
    M = np.zeros((n, n))
    i = 0
    for b in blocks:
        b = np.atleast_2d(np.asarray(b, float))
        M[i:i + b.shape[0], i:i + b.shape[0]] = b
        i += b.shape[0]
    return M

#------------------------------------------------------------------------------

def ins_mekf(x_ins, P_prd, mu, h, Qd, Rd, T_acc, T_ars, imu_meas, m_ref, y_pos=None, y_vel=None):
    """Feedback ESKF (MEKF) for an INS aided by position and a 3-axis magnetometer, unit quaternions with 2 x Gibbs
    vector errors (MSS ins_mekf.m). x_ins = [p, v, b_acc, q, b_ars] (16 states; 17 with Rd['pseudoFlag']), 15 error
    states. Rd is a dict with the keys of the m-file's structure: position, velocity, magnetometer,
    gravityRefVector, pseudoMeasSeaLevel, applicationFlag, pseudoFlag. The magnetometer corrects whenever imu_meas
    has 9 elements."""
    imu_meas = np.ravel(imu_meas)
    pseudo = bool(Rd["pseudoFlag"])
    p, v, b_acc, q = x_ins[0:3], x_ins[3:6], x_ins[6:9], x_ins[9:13]
    Rq = Rquat(q)
    f_imu, w_imu = imu_meas[0:3], imu_meas[3:6]
    rows, dys, Rs = [], [], []
    if y_pos is not None:
        rows.append(np.hstack((I3, O3, O3, O3, O3)))
        dys.append(y_pos - p)
        Rs.append(Rd["position"])
    if y_vel is not None:
        rows.append(np.hstack((O3, I3, O3, O3, O3)))
        dys.append(y_vel - v)
        Rs.append(Rd["velocity"])
    if imu_meas.size >= 9:
        m_imu = imu_meas[6:9]
        v2 = m_imu / vnorm(m_imu)
        v02 = m_ref / vnorm(m_ref)
        rows.append(np.hstack((O3, O3, O3, Smtrx(Rq.T @ v02), O3)))
        dys.append(v2 - Rq.T @ v02)
        Rs.append(Rd["magnetometer"])
    if Rd["applicationFlag"] and imu_meas.size >= 9:
        fg = f_imu - b_acc
        rows.append(np.hstack((O3, O3, O3, Smtrx(Rq.T @ V01), O3)))
        dys.append(fg / vnorm(fg) - Rq.T @ V01)
        Rs.append(Rd["gravityRefVector"])
    if pseudo:
        rows = [np.hstack((r, np.zeros((r.shape[0], 1)))) for r in rows]
        rows.append(np.concatenate((np.zeros(15), [1.0]))[None, :])
        dys.append([-x_ins[16]])
        Rs.append(Rd["pseudoMeasSeaLevel"])
    if dys:
        Cd, dy, Rmat = np.vstack(rows), np.concatenate([np.ravel(d) for d in dys]), _blkdiag(Rs)
    else:
        Cd, dy, Rmat = None, None, None
    p, v, b_acc, q, b_ars, zint, P_hat = _mekf_reset(x_ins, P_prd, Cd, dy, Rmat, pseudo)
    return _mekf_tail(p, v, b_acc, q, b_ars, P_hat, f_imu, w_imu, mu, h, Qd, T_acc, T_ars, pseudo, zint)

#------------------------------------------------------------------------------

def ins_mekf_psi(x_ins, P_prd, mu, h, Qd, Rd, T_acc, T_ars, f_imu, w_imu, y_psi, y_pos=None, y_vel=None):
    """As ins_mekf with a compass (yaw angle y_psi; None when there is no new compass value) instead of the
    magnetometer (MSS ins_mekf_psi.m). The compass row is c_psi = [0, R32, R33]/(R32^2 + R33^2) in the attitude
    columns; the gravity reference vector is used whenever Rd['applicationFlag'] is set."""
    pseudo = bool(Rd["pseudoFlag"])
    p, v, b_acc, q = x_ins[0:3], x_ins[3:6], x_ins[6:9], x_ins[9:13]
    Rq = Rquat(q)
    f_imu, w_imu = np.ravel(f_imu), np.ravel(w_imu)
    rows, dys, Rs = [], [], []
    if y_pos is not None:
        rows.append(np.hstack((I3, O3, O3, O3, O3)))
        dys.append(y_pos - p)
        Rs.append(Rd["position"])
    if y_vel is not None:
        rows.append(np.hstack((O3, I3, O3, O3, O3)))
        dys.append(y_vel - v)
        Rs.append(Rd["velocity"])
    if y_psi is not None:
        c_psi = np.array([0.0, Rq[2, 1], Rq[2, 2]]) / (Rq[2, 1]**2 + Rq[2, 2]**2)
        rows.append(np.concatenate((np.zeros(9), c_psi, np.zeros(3)))[None, :])
        dys.append([ssa(y_psi - math.atan2(Rq[1, 0], Rq[0, 0]))])
        Rs.append(Rd["compass"])
    if Rd["applicationFlag"]:
        fg = f_imu - b_acc
        rows.append(np.hstack((O3, O3, O3, Smtrx(Rq.T @ V01), O3)))
        dys.append(fg / vnorm(fg) - Rq.T @ V01)
        Rs.append(Rd["gravityRefVector"])
    if pseudo:
        rows = [np.hstack((r, np.zeros((r.shape[0], 1)))) for r in rows]
        rows.append(np.concatenate((np.zeros(15), [1.0]))[None, :])
        dys.append([-x_ins[16]])
        Rs.append(Rd["pseudoMeasSeaLevel"])
    if dys:
        Cd, dy, Rmat = np.vstack(rows), np.concatenate([np.ravel(d) for d in dys]), _blkdiag(Rs)
    else:
        Cd, dy, Rmat = None, None, None
    p, v, b_acc, q, b_ars, zint, P_hat = _mekf_reset(x_ins, P_prd, Cd, dy, Rmat, pseudo)
    return _mekf_tail(p, v, b_acc, q, b_ars, P_hat, f_imu, w_imu, mu, h, Qd, T_acc, T_ars, pseudo, zint)

#------------------------------------------------------------------------------

def ins_heave(x_ins, P_prd, h, Qd, Rd, T_acc, mu, rho, f_imu, phi, theta, p_0=None, p=None, exact_fD=False):
    """Feedback ESKF for heave aided by a pressure sensor (MSS ins_heave.m). x_ins = [z, w, b_acc_z]. The down
    component of the specific force is computed as in the m-file, f_D = -f_x cos(phi) sin(theta) + f_y sin(phi)
    + f_z cos(phi) cos(theta). The corrector runs when the pressure p is given (depth (p - p_0)/(rho g)).
    exact_fD=True (not in MSS): f_D from the third row of Rzyx,
    -f_x sin(theta) + f_y cos(theta) sin(phi) + f_z cos(theta) cos(phi)."""
    g = gravity(mu)
    A = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0 / T_acc]])
    Ad = expm(A * h)
    Cd = np.array([[1.0, 0.0, 0.0]])
    Ed = h * np.array([[0.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    f_imu = np.ravel(f_imu)
    z, w, b = x_ins
    if exact_fD:
        f_D = -f_imu[0] * math.sin(theta) + f_imu[1] * math.cos(theta) * math.sin(phi) \
            + f_imu[2] * math.cos(theta) * math.cos(phi)
    else:
        f_D = -f_imu[0] * math.cos(phi) * math.sin(theta) + f_imu[1] * math.sin(phi) \
            + f_imu[2] * math.cos(phi) * math.cos(theta)
    if p is None:
        P_hat = P_prd
    else:
        K = P_prd @ Cd.T / (Cd @ P_prd @ Cd.T + Rd)
        IKC = np.eye(3) - K @ Cd
        eps_z = (p - p_0) / (rho * g) - z
        dx = (K * eps_z).ravel()
        P_hat = IKC @ P_prd @ IKC.T + Rd * (K @ K.T)
        z, w, b = z + dx[0], w + dx[1], b + dx[2]
    P_prd = Ad @ P_hat @ Ad.T + Ed @ Qd @ Ed.T
    a_z = f_D - b + g
    z = z + h * w + 0.5 * h**2 * a_z
    w = w + h * a_z
    return np.array([z, w, b]), P_prd
