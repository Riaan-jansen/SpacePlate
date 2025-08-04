import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from scipy.special import jv, j0, j1  # bessel function 1st kind

# minimiser tool - is it just differences?
from scipy.optimize import minimize

from modal_match import T_DF

c = 343000
f_target = 12800  # (Hz)
k = f_target * 2*pi / c
T_target = 0.2

def optimiser(params, k, modes):
    h, hg = params

    # T_DF returns array len(k) if k is scalar so should T_DF
    T_val = T_DF(k, modes, h, hg)

    T_abs = np.abs(T_val)
    return np.abs(T_abs - T_target)
