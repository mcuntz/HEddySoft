#!/usr/bin/env python
"""
Find covariance peak in lagged covariance series

This module was written by Bernard Heinesch, University of Liege,
Gembloux Agro-Bio Tech, adapted by Matthias Cuntz while at Institut
National de Recherche pour l'Agriculture, l'Alimentation et
l'Environnement (INRAE), Nancy, France.

:copyright: Copyright 2025- Bernard Heinesch, Matthias Cuntz, see AUTHORS.rst for details.
:license: MIT License, see LICENSE for details.

.. moduleauthor:: Matthias Cuntz

The following functions are provided:

.. autosummary::
   find_covariance_peak

History
   * Written, 2025, Bernard Heinesch
   * outer_window = 5 * inner_window, Sep 2026, Matthias Cuntz
   * removed outer_window, Oct 2026, Matthias Cuntz

"""
import numpy as np
from scipy.signal import filtfilt, argrelmin, argrelmax
import matplotlib.pyplot as plt


__all__ = ['find_covariance_peak']


def find_covariance_peak(covariance, c_default, c_min, c_max,
                         filter_length, positive_peak_only=False,
                         plot_label=tuple()):
    """
    For two shifted (anti)correlated signals, s1 and s2,
    this fuction estimates the lag of s2 with respect to s1 by finding the
    maximum/minimum of the (smoothed) crosscovariance xcov(s1, s2)

    Parameters
    ----------
    covariance : array-like
        Covariance or correlation function
    c_default : int
        Default lag sample if no extreme found
    c_min : int
        Minimum lag sample to search timelag
    c_max : int
        Maximum lag sample to search timelag
    filter_length : int
        Smoothing length in samples of filter applied prior to
        finding covariance peak
    positive_peak_only : bool, optional
        If True, only the positive peak in the cov function will be searched
    plot_label: tuple, optional
        Filebase and Label used when plotting the covariance peak search

    Returns
    -------
    int
        lag in samples

    Notes
    -----
    - if filter_lenght is too high, an error will occur in the filtfilt function
    - does not follow the strategy of InnFlux (with derivatives),
      which was found to be inadequate with low SNR

    """
    # filter parameters
    if filter_length != 1:
        b = np.hamming(filter_length) / sum(np.hamming(filter_length))
        a = 1
    if positive_peak_only:
        if filter_length != 1:
            filtered_cov = filtfilt(b, a, covariance.astype(float), axis=0,
                                    padtype='odd',
                                    padlen=3*(max(len(b), len(a))-1))
        else:
            # no filtering
            filtered_cov = covariance
        ii = argrelmax(filtered_cov)[0]
    else:
        if filter_length != 1:
            filtered_cov = filtfilt(b, a, covariance.astype(float))
        else:
            # no filtering
            filtered_cov = covariance

        ii1 = argrelmax(filtered_cov)[0]
        ii2 = argrelmin(filtered_cov)[0]
        if (len(ii1) > 0) or (len(ii2) > 0):
            ii = np.append(ii1, ii2)
        else:
            ii = []

    if len(ii) == 0:
        idx = c_default - c_min
        lag_samples = c_default
    elif len(ii) == 1:
        idx = ii[0]
        lag_samples = c_min + idx
    else:
        cc = c_min + ii
        # if several extremes, take the extreme closest to the default value
        indx = np.argmin(np.abs(cc - c_default))
        lag_samples = cc[indx]
        idx = lag_samples - c_min
    maxcov = covariance[idx]

    # plot results if requested
    if len(plot_label) > 0:

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # plot whole cov function and found maximum
        c_size = c_max - c_min + 1
        xxrange = np.linspace(c_min, c_max, num=c_size)
        ax.plot(xxrange, covariance, color='blue')
        ax.plot(xxrange, filtered_cov, color='orange')
        ax.set_ylabel('Cov (uncal units)')
        ax.set_xlabel('Samples')

        ax.set_title('Lag determination for: ' + plot_label[1])

        ax.axvline(x=lag_samples, color='black',
                   label='axvline - full height')
        ax.plot(lag_samples, maxcov,
                marker="o", markersize=8, markeredgecolor="red",
                markerfacecolor="red")

        plt.tight_layout()

        # plt.show(block=False)
        plt.savefig(f'{plot_label[0]}-covariance_{plot_label[1]}.png')
        plt.close()  # close all previous plots
        # plt.pause(1)  # pauses one second before disappearing
        # plt.close('all')  # close all previous plots

    return maxcov, int(lag_samples)
