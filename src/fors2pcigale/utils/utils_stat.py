""" Module to provide simple calculation in statitics"""

# pylint: disable=line-too-long
# pylint: disable=trailing-newlines
# pylint: disable=redundant-u-string-prefix
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=unused-import
# pylint: disable=anomalous-backslash-in-string
# pylint: disable=too-many-locals
# pylint: disable=broad-exception-caught
# pylint: disable=too-many-statements
# pylint: disable=trailing-whitespace
# pylint: disable=no-else-return

import numpy as np


def weighted_mean(var, wts=None):
    """Calculates the weighted mean"""
    return np.average(var, weights=wts)


def weighted_variance(var, wts=None):
    """Calculates the weighted variance"""

    N = len(var)

    if N == 0:
        var = np.zeros([0.])
        wts = np.ones(1)

    if wts is None:
        wts = np.ones(N)

    if len(var) == 1:
        return 1/wts[0]
    else:
        return np.average((var - weighted_mean(var, wts))**2, weights=wts)


def weighted_skew(var, wts):
    """Calculates the weighted skewness"""
    return (np.average((var - weighted_mean(var, wts))**3, weights=wts) /
            weighted_variance(var, wts)**(1.5))

def weighted_kurtosis(var, wts):
    """Calculates the weighted skewness"""
    return (np.average((var - weighted_mean(var, wts))**4, weights=wts) /
            weighted_variance(var, wts)**(2))
