from collections import defaultdict
import numpy as np
import scipy
import scipy.sparse as sps
import math
import matplotlib.pyplot as plt
import time

def soft_threshold(internal, reg_coef, current_feature_norm):
    if internal > reg_coef:
        return (internal - reg_coef) / (current_feature_norm ** 2)
    elif internal < - reg_coef:
        return (internal + reg_coef) / (current_feature_norm ** 2)
    return 0.0

def add_unit_feature(A):
    ones = np.ones((np.shape(A)[0], 1))
    if not sps.issparse(A):
        return np.hstack([ones, A])
    return sps.hstack([ones, A], format="csr")
