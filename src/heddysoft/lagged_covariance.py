#!/usr/bin/env python
"""
Lagged covariance

This module was written by Bernard Heinesch, University of Liege,
Gembloux Agro-Bio Tech, adapted by Matthias Cuntz while at Institut
National de Recherche pour l'Agriculture, l'Alimentation et
l'Environnement (INRAE), Nancy, France.

:copyright: Copyright 2025- Bernard Heinesch, Matthias Cuntz, see AUTHORS.rst for details.
:license: MIT License, see LICENSE for details.

.. moduleauthor:: Matthias Cuntz

The following functions are provided:

.. autosummary::
   cov
   lagged_covariance
   xcov

History
   * Written, 2025, Bernard Heinesch
   * New cov function, Sep 2026, Matthias Cuntz
   * Added and adapted compute_time_lag function as lagged_covariance,
     Sep 2026, Matthias Cuntz
   * Pass filebase to lagged_covariance, Oct 2026, Matthias Cuntz

"""
import numpy as np
from scipy.signal import correlate, convolve
from .config import read_config
from .find_covariance_peak import find_covariance_peak


__all__ = ['cov', 'xcov', 'lagged_covariance']


def cov(df, xname, yname):
    """
    Covariance between xname and yname in dataframe df

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with xname and yname columns
    xname : str
        1st column name
    yname : str
        2nd column name

    Returns
    -------
    corr
        covariance

    """
    return df[[xname, yname]].cov(ddof=1).loc[xname, yname]


def xcov(x, y, lag):
    """
    Lagged covariances between lag interval

    Parameters
    ----------
    x : array_like
        1st signal
    y : array_like
        2nd signal
    lag : iterable
        Lag interval [start, end] in samples

    Returns
    -------
    corr
        Lagged covariances

    Notes
    -----
    - not influenced by circular behaviour of the roll operation
    - when translating from MATLAB (InnFlux), no xcov function was found
      in python packages
    - MATLAB computes the mean of the initial array while python (and excel)
      uses the mean of the truncated arrays. So gives similar results for a
      zero lag but different results for a non-zero lag!

    """
    lag = np.linspace(lag[0], lag[1], num=lag[1] - lag[0] + 1, dtype='int16')

    # create output array
    crosscov = np.full(len(lag), np.nan)

    x = np.asarray(x)
    y = np.asarray(y)
    for i in range(0, len(lag), 1):
        ll = int(lag[i])
        yshifted = np.roll(y, ll)

        if ll > 0:
            xs = x[ll:]
            ys = yshifted[ll:]
        else:
            xs = x[0:len(x) - 1 - ll]
            ys = yshifted[0:len(x) - 1 - ll]

        mask = np.isfinite(xs) & np.isfinite(ys)
        n = int(mask.sum())
        if n < 2:
            crosscov[i] = np.nan
            continue

        xs = xs[mask]
        ys = ys[mask]
        # n/(n-1)*(E(x*y)-E(x)E(y))
        crosscov[i] = (np.sum(xs * ys) - (np.sum(xs) * np.sum(ys) / n)) / (n - 1)

    # reduce to float if start = end lag
    if len(crosscov) == 1:
        crosscov = crosscov[0]

    return crosscov


def lagged_covariance(species, config, w_prime, c_prime, timelag_dict,
                      rh=-9999, plot_label=tuple()):
    """
    Compute optimal time lag between vertical wind and a scalar signal

    Parameters
    ----------
    species : str
        co2 or h2o
    config : str or astConfigParser
        Config (filename)
    w_prime : array_like
        Vertical wind fluctuations
    c_prime: np.ndarray
        Scalar fluctuations of species
    timelag_dict : dict
        Dictionary with appropriate information for timelag detection
    rh: float, optional
        Relative humidity (0-1) for eddypro option
    plot_label: tuple, optional
        Filebase and Label used when plotting the covariance peak search

    Returns
    -------
    wc, lag_samples : np.array, int
        Cross-covariance between w_rpime and lagged c_prime;
        Lag in c_prime

    Notes
    -----
    The function implements the lag logic used in GEddySoft:
        max: extreme of lag, smoothed according to filter_length, within index window
            [win_center - win_size, win_center + win_size]
        eddypro: max with default values from EddyPro timelag file
            with relative humidity classes for H2O timelag
        file: read timelags for CO2 and H2O from file
            missing timelags in seconds will be linearily interpolated and
            missing timelags indexes will be interpolated ny nearest neighbours

    """
    if plot_label == '':
        plot_label = species

    cfg = read_config(config)

    lag_method = cfg.get('timelag', 'method')
    if lag_method.lower() != timelag_dict['method']:
        raise ValueError(f'Timelag method of config file ({lag_method})'
                         f' does not fit timelag dictionary'
                         f' ({timelag_dict["method"]})')

    if lag_method.lower() == 'file':
        lags = timelag_dict['timelags']  # DataFrame with prescribed lags
        # closest to current c_prime
        idx = np.argmin(np.abs(lags.index - c_prime.index[-1]))
        itimelag = lags[f'{species.lower()}_timelag_index'].iloc[idx]
        # covariance
        wc = xcov(w_prime, c_prime, [itimelag, itimelag])

        return wc, itimelag

    elif lag_method.lower() == 'eddypro':
        sampling_rate = cfg.get('sonic', 'sampling_rate')
        lags = timelag_dict[f'{species.lower()}_timelag']
        if species.lower() == 'co2':
            c_center = round(lags[0] * sampling_rate)
            c_min = round(lags[1] * sampling_rate)
            c_max = round(lags[2] * sampling_rate)
        elif species.lower() == 'h2o':
            relh = rh * 100.  # %
            if relh < -10.:
                raise ValueError('Relative humidity has to given for'
                                 ' timelag determination with EddyPro file')
            idx = np.argmin(np.abs(lags['rh'] - relh))
            c_center = round(lags['h2omed'].iloc[idx] * sampling_rate)
            c_min = round(lags['h2omin'].iloc[idx] * sampling_rate)
            c_max = round(lags['h2omax'].iloc[idx] * sampling_rate)
        else:
            raise ValueError(f'Species not known: {species}')
        c_size = c_max - c_min + 1
        c_filter_length = cfg.get('timelag', f'{species.lower()}_filter_length')

        # cov_wc = xcov(w_prime, c_prime, [c_min, c_max])
        # maxcov, itimelag = find_covariance_peak(cov_wc,
        #                                         c_center, c_min, c_max,
        #                                         c_filter_length, 0, plot_label)

        m1 = len(c_prime) - 1  # zero lag
        # correlate is about 4 times faster than xcov
        corr = correlate(w_prime, c_prime)[m1 + c_min:m1 + c_max + 1]
        _, itimelag = find_covariance_peak(corr, c_center, c_min, c_max,
                                           c_filter_length,
                                           plot_label=plot_label)
        maxcov = xcov(w_prime, c_prime, [itimelag, itimelag])

        return maxcov, itimelag

    elif lag_method.lower() == 'max':
        c_center = timelag_dict[f'{species.lower()}_center']
        c_size = timelag_dict[f'{species.lower()}_size']
        c_filter_length = cfg.get('timelag', f'{species.lower()}_filter_length')
        c_min = c_center - c_size
        c_max = c_center + c_size

        # cov_wc = xcov(w_prime, c_prime, [c_min, c_max])
        # maxcov, itimelag = find_covariance_peak(cov_wc,
        #                                         c_center, c_min, c_max,
        #                                         c_filter_length, 0, plot_label)

        m1 = len(c_prime) - 1  # zero lag
        # correlate is about 4 times faster than xcov
        corr = correlate(w_prime, c_prime)[m1 + c_min:m1 + c_max + 1]
        _, itimelag = find_covariance_peak(corr, c_center, c_min, c_max,
                                           c_filter_length,
                                           plot_label=plot_label)
        maxcov = xcov(w_prime, c_prime, [itimelag, itimelag])

        return maxcov, itimelag

    else:
        raise ValueError(f'Lag method not known: {lag_method}')
