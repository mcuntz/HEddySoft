#!/usr/bin/env python
"""
Cospectrum

This module was written by Bernard Heinesch, University of Liege,
Gembloux Agro-Bio Tech, adapted by Matthias Cuntz while at Institut
National de Recherche pour l'Agriculture, l'Alimentation et
l'Environnement (INRAE), Nancy, France.

:copyright: Copyright 2025- Bernard Heinesch, Matthias Cuntz, see AUTHORS.rst for details.
:license: MIT License, see LICENSE for details.

.. moduleauthor:: Matthias Cuntz

The following functions are provided:

.. autosummary::
   cospectrum

History
   * Written, Dec 2022, Bernard Heinesch
   * Added spectrum, July 2024, Bernard Heinesch
   * ini.get(), Sep 2026, Matthias Cuntz

"""
import matplotlib.pyplot as plt
import numpy as np
from .log_bin_spectrum import log_bin_spectrum
from .kaimal_cospectrum import kaimal_cosp


__all__ = ['cospectrum']


def cospectrum(cfg, x, y, f, mean_wind_speed,
               spectrum_type='', plot='', tag=''):
    """
    Computes (co)spectrum of two variables, from InnFlux code
    for spectrum, simply input the same time series for x and y

    Parameters
    ----------
    cfg : dict
        initialisation information
    x : numpy arr of float
        fluctuations of the first time series
    y : numpy arr of float
        fluctuations of the second time series
    f : numpy arr of float
        frequency axis of cospectra [s-1]
    mean_wind_speed: float
        mean wind speed [m s-1]
    spectrum_type : str, optional
        text to be used in the plot ('spec' or 'cospec')
    plot : str, optional
        If not empty, plot cospectrum with filename starting with plot
    tag : str, optional
        Tag for filename and plto title

    Returns
    -------
    cospec_xy: numpy arr of float, (co)spectrum for xy, Co(x',y',f)
    cospec_xy_scaled: numpy arr of float, scaled (co)spectrum f*Co(x',y',f)/xy
    cospec_xy_integral: numpy arr of float, integral of (co)spectrum

    """
    nsoll = cfg.get('sonic', 'file_length') * cfg.get('sonic', 'sampling_rate')
    nbins = cfg.get('param', 'num_freq_bins')
    sonic_height = cfg.get('sonic', 'sensor_height')
    # two-sided cospec
    cospec2 = np.real(np.conj(np.fft.fft(x)) / nsoll * np.fft.fft(y) / nsoll)
    # extract the lower ~1/2 of the spec
    cospec = np.copy(cospec2[0:len(f)])
    # multiply all but the first and the last spectral entry by 2
    cospec[1:len(cospec)-2] = 2 * cospec[1:len(cospec)-2]
    # scale by delta_f
    cospec = cospec / (f[1] - f[0])
    # compute integral (for normalisation purposes)
    cospec_xy_integral = np.trapezoid(cospec, f)

    # cospectrum log-bin-averaged in regular f domain
    f_bin, cospec_xy = log_bin_spectrum(f, cospec, nbins, f[1], f[-1])
    # normalized, f-scaled log-bin-averaged cospectrum
    cospec_xy_scaled = f_bin * cospec_xy / cospec_xy_integral

    # additional comments:
    # - first value of cospec2 is the mean. In this case =zero because fluctuations are provided as input
    # - multiplication by 2 is done for retreiving the energy content (Parsival theorem)
    # - with a discrete approach, the covariance can also be calculated as the
    #   sum of cospec2 and the sum of cospec before scaling by delta f)

    # plot results if requested
    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(8, 10))

        # plot f*cospec in log-log scales
        ax[0].scatter(f, f*cospec, s=10, color='grey', label='all')
        ax[0].plot(f_bin, f_bin*cospec_xy, color='blue', label='binned')
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[0].set_ylabel(f'f*{spectrum_type} (uncal units)')
        ax[0].set_xlabel('natural f [Hz]')

        if tag:
            ax[0].set_title(f'{spectrum_type} for {tag}')
        else:
            ax[0].set_title(f'{spectrum_type}')

        # creation of the normalized frequency scale, for ploting purposes only
        fnorm = f * sonic_height / mean_wind_speed
        fnorm_bin = f_bin * sonic_height / mean_wind_speed

        # plot f*(co)spec/cov in normalized frequency scale and log-log scales
        ax[1].scatter(fnorm, f*cospec/cospec_xy_integral, s=10, color='grey', label='all')
        ax[1].plot(fnorm_bin, cospec_xy_scaled, color='red', label='binned')
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        if spectrum_type == 'cospec':
            ax[1].set_ylabel(f'f*{spectrum_type}/cov (uncal units)')
        else:
            ax[1].set_ylabel(f'f*{spectrum_type}/var (uncal units)')
        ax[1].set_xlabel('normalized f [-]')

        # add (unstable) Kaimal in case of cospectrum
        if spectrum_type == 'cospec':
            # compute and plot Kaimal cospectrum
            kaimal = np.zeros(len(f_bin))
            for x in range(1, len(f_bin)):
                kaimal[x] = kaimal_cosp(f_bin[x], f_bin[x] * sonic_height / mean_wind_speed, -0.001)
            ax[1].plot(fnorm_bin, f_bin*kaimal, color='black', label='Kaimal)')
            ax[1].legend(loc="lower left")

        # plt.show(block=False)
        if tag:
            plt.savefig(f'{plot}-{spectrum_type}_{tag}.png')
        else:
            plt.savefig(f'{plot}-{spectrum_type}.png')
        plt.close()

    return [cospec_xy, cospec_xy_scaled, cospec_xy_integral]
