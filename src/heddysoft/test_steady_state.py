#!/usr/bin/env python
"""
Steady-state tests

This module was written by Bernard Heinesch, University of Liege,
Gembloux Agro-Bio Tech, adapted by Matthias Cuntz while at Institut
National de Recherche pour l'Agriculture, l'Alimentation et
l'Environnement (INRAE), Nancy, France.

:copyright: Copyright 2025- Bernard Heinesch, Matthias Cuntz, see AUTHORS.rst for details.
:license: MIT License, see LICENSE for details.

.. moduleauthor:: Matthias Cuntz

The following functions are provided:

.. autosummary::
   test_steady_state_D99
   test_steady_state_FW96
   test_steady_state_M98


History
   * Written Foken & Wichura (1996) test and Mahrt (1998) test,
     Oct 2022, Bernard Heinesch
   * Written Dutaur (1999) and Nemitz (2002) test, April 2023, D. Tzvetkov

"""
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from .lagged_covariance import xcov


__all__ = ['test_steady_state_D99', 'test_steady_state_FW96',
           'test_steady_state_M98']


def test_steady_state_D99(w_prime, c_prime, plot=False):
    """
    Performs the non-stationary test described by Dutaur (1999) and Nemitz (2002)

    Parameters
    ----------
    x : array-like
        raw high-frequency time series of vertical wind component (i.e. W)
    y : array-like
        raw high-frequency time series of scalar atmospheric variable (e.g. CO2)
    plot : bool, optional
        If True, plot test

    Returns
    -------
    Returns the relative covariance stationarity criterion.

    """
    # cumulated sum of the product
    F_T = np.cumsum(w_prime*c_prime) / len(w_prime)

    t = np.linspace(0, len(w_prime), num=len(w_prime), dtype=int)

    # linear fit
    slope, intercept, r_value, p_value, std_mean = stats.linregress(t, F_T)

    # compute the residuals
    fitted_values = slope * t + intercept
    residuals = F_T - slope * t + intercept
    
    # compute the standard deviation of the residuals
    stddev = residuals.std()

    # compute the relative covariance stat criterion
    RCSC = abs(2 * stddev / np.nanmean(w_prime * c_prime))

    # plot results if requested
    if plot:
        plt.figure(dpi=400)
        plt.title('Dutaur 1999 stationarity test')
        plt.xlabel('time')
        plt.ylabel('cum covar')
        plt.plot(t, F_T, label='data')
        plt.plot(t, t*slope + intercept, label='lin reg')
        plt.legend(title=("sigma residuals = " +
                          str('{:.2e}'.format(np.round(stddev, 7))) + '\n' +
                          'RCSC = '+ str('{:.2e}'.format(np.round(RCSC, 6)))))

    return RCSC


def test_steady_state_FW96(w_prime, c_prime):
    """
    Performs the steady state test for w'c' according to Foken & Wichura (1996)
    see Chapter 9 in X. Lee et al. (eds.), Handbook of Micrometeorology (2004)

    Parameters
    ----------
    w_prime : array_like
        raw high-frequency time series of vertical wind component (i.e. W)
    c_prime : array_like
        raw high-frequency time series of scalar atmospheric variable (e.g. CO2)

    Returns
    -------
    Returns the non-stationary test statistic.

    """
    subset_size = math.floor(len(w_prime) / 5)
    wc_subset = np.zeros(5)
    for j in range(1, 6):
        wc_subset[j-1] = xcov(w_prime[(j-1) * subset_size:j * subset_size],
                              c_prime[(j-1) * subset_size:j * subset_size],
                              [0, 0])

    wc_total = xcov(w_prime, c_prime, [0, 0])
    S_FW96 = abs((np.nanmean(wc_subset) - wc_total) / wc_total)

    return S_FW96


def test_steady_state_M98(x, y):
    """
    Performs the non-stationary ratio test described by Mahrt (1998)
    after Vitale et al. 2020, Biogeosciences, "Robust data cleaning procedure
    for eddy covariance flux measurements"" and associated inst_prob_test
    routine found on Gitlab

    Parameters
    ----------
    x : array-like
        raw high-frequency time series of vertical wind component (i.e. W)
    y : array-like
        raw high-frequency time series of scalar atmospheric variable (e.g. CO2)

    Returns
    -------
    Returns the non-stationary ratio test statistic.

    Notes
    -----
    - computation of covariances might be more efficient using rolling and cov functions
      on df  (e.g. https://www.geeksforgeeks.org/how-to-calculate-rolling-correlation-in-python/),
      or the np.roll function

    """
    # compute COV, covariance on the whole dataset
    COV = xcov(x, y, [0, 0])

    # compute COVs, covariances over 6 intervals
    subset_size = math.floor(len(x) / 6)
    COVs = np.zeros(6)
    for j in range(1, 7):
        COVs[j-1] = xcov(x[(j-1) * subset_size:j * subset_size - 1],
                         y[(j-1) * subset_size:j * subset_size - 1],
                         [0, 0])

    # compute COVw, covariances over 6*6 intervals
    subset_size = math.floor(len(x) / 36)
    COVw = np.zeros(36)
    for j in range(1, 37):
        COVw[j-1] = xcov(x[(j-1) * subset_size:j * subset_size - 1],
                         y[(j-1) * subset_size:j * subset_size - 1],
                         [0, 0])

    sigmaB = np.sqrt(np.nansum((COVs - COV)**2) / (len(COVs) - 1))

    sigmaW1 = np.sqrt(1 / 5 * np.nansum((COVw[0:5] - COVs[0])**2)) if len(COVw) >= 6 else np.nan
    sigmaW2 = np.sqrt(1 / 5 * np.nansum((COVw[6:11] - COVs[1])**2)) if len(COVw) >= 12 else  np.nan
    sigmaW3 = np.sqrt(1 / 5 * np.nansum((COVw[12:17] - COVs[2])**2)) if len(COVw) >= 18 else np.nan
    sigmaW4 = np.sqrt(1 / 5 * np.nansum((COVw[18:23] - COVs[3])**2)) if len(COVw) >= 24 else np.nan
    sigmaW5 = np.sqrt(1 / 5 * np.nansum((COVw[24:29] - COVs[4])**2)) if len(COVw) >= 30 else np.nan
    sigmaW6 = np.sqrt(1 / len(COVw) * np.nansum((COVw[30:len(COVw)]-COVs[5])**2)) if (len(COVw) > 30 and len(COVw) <= 36) else np.nan
    sigmaWi = [sigmaW1, sigmaW2, sigmaW3, sigmaW4, sigmaW5, sigmaW6]
    sigmaW = np.nanmean(sigmaWi)

    # compute the non-stationarity ratio
    S_M98 = sigmaB / (sigmaW / np.sqrt(np.nansum(~np.isnan(sigmaWi))))

    return S_M98
