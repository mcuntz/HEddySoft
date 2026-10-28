#!/usr/bin/env python
"""
Turbulent flux uncertainties

This module was written by Bernard Heinesch, University of Liege,
Gembloux Agro-Bio Tech, adapted by Matthias Cuntz while at Institut
National de Recherche pour l'Agriculture, l'Alimentation et
l'Environnement (INRAE), Nancy, France.

:copyright: Copyright 2025- Bernard Heinesch, Matthias Cuntz, see AUTHORS.rst for details.
:license: MIT License, see LICENSE for details.

.. moduleauthor:: Matthias Cuntz

The following functions are provided:

.. autosummary::
   flux_uncertainties

History
   * Written, Nov 2022, Bernard Heinesch

"""
import numpy as np
from .config import read_config
from .lagged_covariance import xcov


__all__ = ['flux_uncertainties']


def flux_uncertainties(config, w_prime, c_prime):
    """
    Different forms of turbulent flux uncertainties

    Rarameters
    ----------
    config : str or astConfigParser
        Config (filename)
    w_prime : array_like
        Vertical wind fluctuations
    c_prime: np.ndarray
        Scalar fluctuations of species

    Returns
    -------
    flux_noise_mean, flux_noise_std, flux_noise_rmse, random_error_FS,
    random_flux, random_error_noise : list of floats
        flux_noise_mean: flux detection limit using mean of covariance
                         between +/- 160-180s (Spirig et al. 2005)
        flux_noise_std: flux detection limit using STD of covariance
                        between +/- 160-180s (Spirig et al. 2005)
        flux_noise_rmse: flux detection limit: flux noise criterium
                         using RMSE noise of covariance between +/- 160-180s
        random_error_FS: random error as described by Finkelstein and Sims 2001
        random_flux: flux detection limit: random shuffle criterium
                     as described by Billesbach 2011
        random_error_noise: random error from white noise using
                            autocovariance (Lenschow 2000, Mauder 2013,
                            Langford 2015)

    Notes
    -----
    For the white noise computation, modification compared to MATLAB Innflux:
        sqrt of the difference instead of difference of the sqrt
        (cf. Langford 2015 p. 4200f)
    warning: in the MATLAB Innflux version, interpolated variance at zero lag
             can be negative and the MATLAB sqrt gives an imaginary number

    """
    cfg = read_config(config)

    sampling_rate = cfg.get('sonic', 'sampling_rate')
    file_length = cfg.get('sonic', 'file_length')
    nsoll = int(file_length * sampling_rate)

    # flux detection limit:
    # flux noise criterium using STD noise of covariance between +/- 160-180s
    # (Spirig et al. 2005)
    cov_wc_left = xcov(w_prime, c_prime,
                       [int(-180 * sampling_rate), int(-160 * sampling_rate)])
    cov_wc_right = xcov(w_prime, c_prime,
                        [int(160 * sampling_rate), int(180 * sampling_rate)])
    flux_noise_mean = np.nanmean(np.concatenate((cov_wc_left, cov_wc_right)))
    flux_noise_std = np.nanstd(np.concatenate((cov_wc_left, cov_wc_right)),
                               ddof=1)
    flux_noise_rmse = np.sqrt( 0.5*((np.nanstd(cov_wc_left, ddof=1))**2 +
                                    (np.nanmean(cov_wc_left))**2 +
                                    (np.nanstd(cov_wc_right, ddof=1))**2 +
                                    (np.nanmean(cov_wc_right))**2) )

    # estimate random error as described by Finkelstein and Sims 2001
    random_error_FS = np.sqrt(1. / nsoll *
                              (sum(xcov(w_prime, w_prime, [0, 99]) *
                                   xcov(c_prime, c_prime, [0, 99])) +
                               sum(xcov(w_prime, c_prime, [0, 99]) *
                                   xcov(c_prime, w_prime, [0, 99])) ) )

    # flux detection limit:
    # random shuffle criterium as described by Billesbach 2011
    # np.random.seed(1234)  # for reproducibility tests
    xcov_rand = np.empty(10)
    for irand in range(0, 10):
        w_rand = w_prime[np.random.permutation(len(w_prime))]
        xcov_rand[irand] = xcov(w_rand, c_prime, [0, 0])
    random_flux = np.nanstd(xcov_rand)

    # estimate white noise using autocovariance
    # (Lenschow 2000, Mauder 2013, Langford 2015)
    autocov_c = xcov(c_prime, c_prime, [0, 4])
    x_fit = np.arange(0, 4, dtype=float)
    y_fit = np.asarray(autocov_c[0:4], dtype=float)
    mask = np.isfinite(x_fit) & np.isfinite(y_fit)
    if np.isfinite(autocov_c[4]) and int(mask.sum()) >= 2:
        try:
            p = np.polyfit(x_fit[mask], y_fit[mask], 1)
            y4 = np.polyval(p, 4)
            w2 = autocov_c[4] - y4
            if w2 >= 0:
                white_noise_c = np.sqrt(autocov_c[4] - y4)
            else:
                white_noise_c = np.nan
        except np.linalg.LinAlgError:
            white_noise_c = np.nan
    else:
        white_noise_c = np.nan    
    random_error_noise = (white_noise_c * np.nanstd(w_prime, ddof=1) /
                          np.sqrt(nsoll))

    return [flux_noise_mean, flux_noise_std, flux_noise_rmse,
            random_error_FS, random_flux, random_error_noise]
