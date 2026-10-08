#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guidance algorithms.

Reference: T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and
Motion Control. 2nd. Edition, Wiley. 
URL: www.fossen.biz/wiley

Author:     Thor I. Fossen
"""

import numpy as np
import math
from .gnc import ssa

# [x_d,v_d,a_d] = refModel3(x_d,v_d,a_d,r,wn_d,zeta_d,v_max,sampleTime) is a 3-order 
# reference  model for generation of a smooth desired position x_d, velocity |v_d| < v_max, 
# and acceleration a_d. Inputs are natural frequency wn_d and relative damping zeta_d.
def refModel3(x_d, v_d, a_d, r, wn_d, zeta_d, v_max, sampleTime):
    
    # desired "jerk"
    j_d = wn_d**3 * (r -x_d) - (2*zeta_d+1) * wn_d**2 * v_d - (2*zeta_d+1) * wn_d * a_d

   # Forward Euler integration
    x_d += sampleTime * v_d             # desired position
    v_d += sampleTime * a_d             # desired velocity
    a_d += sampleTime * j_d             # desired acceleration 
    
    # Velocity saturation
    if (v_d > v_max):
        v_d = v_max
    elif (v_d < -v_max): 
        v_d = -v_max    
    
    return x_d, v_d, a_d

#------------------------------------------------------------------------------

def refModel(x_d, v_d, a_d, x_ref, v_max, zeta_d, w_d, h, euler_angle=True):
    """Third-order reference model (MSS refModel.m): desired position, velocity and acceleration at t + h."""
    e_x = ssa(x_d - x_ref) if euler_angle else x_d - x_ref
    a_d_dot = -w_d**3 * e_x - (2 * zeta_d + 1) * w_d**2 * v_d - (2 * zeta_d + 1) * w_d * a_d
    x_d, v_d, a_d = x_d + h * v_d, v_d + h * a_d, a_d + h * a_d_dot
    if abs(v_d) > v_max:
        v_d = np.sign(v_d) * v_max
    return x_d, v_d, a_d
