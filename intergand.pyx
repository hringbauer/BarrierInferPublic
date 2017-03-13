import numpy as np


def integrand_c(t, r, nbh, L):
    '''Integrand for numerical integration'''
    res = 1 / (2 * t * nbh) * np.exp(-(r ** 2) / (4 * t) - L * t)
    return res
