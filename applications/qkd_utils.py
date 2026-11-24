"""Shared channel model and distance solver for the Figure 7 experiments.

This module contains numerical helpers only; run qkd_distance.py or repeater.py
for the paper figures.
"""

import numpy as np

ed = 0.015
Y0 = 3 * (10**-8)
e0 = 1/2
f = 1.06

def H(x):
    return -x*np.log2(x) - (1-x)*np.log2(1-x)

def calcETA(l ,alpha=0.21):
    return 10**(-l*alpha/10)

def calcE1(l, ed):
    eta = calcETA(l)
    e1 = (eta*ed + (1-eta)*Y0*e0) / (eta + (1-eta) * Y0)
    return e1

def find_zero_distance(rate_func, ed=0.015, tol=1e-6, max_iter=60, *args, **kwargs):
    """Binary-search for the distance where rate_func returns 0 between low and high."""
    low = 50
    high = 500
    f_low = rate_func(low, ed, *args, **kwargs)
    f_high = rate_func(high, ed, *args, **kwargs)
    if f_low == 0:
        return low
    if f_high == 0:
        return high

    if f_low * f_high > 0:
        raise ValueError("rate_func must have opposite signs at the interval endpoints")

    for _ in range(max_iter):
        mid = (low + high) / 2
        f_mid = rate_func(mid, ed, *args, **kwargs)
        if abs(f_mid) <= tol:
            return mid
        if f_low * f_mid <= 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid

    return (low + high) / 2
