#!/usr/bin/env python
"""
Log-bin average

This module was written by Bernard Heinesch, University of Liege,
Gembloux Agro-Bio Tech, adapted by Matthias Cuntz while at Institut
National de Recherche pour l'Agriculture, l'Alimentation et
l'Environnement (INRAE), Nancy, France.

:copyright: Copyright 2025- Bernard Heinesch, Matthias Cuntz, see AUTHORS.rst for details.
:license: MIT License, see LICENSE for details.

.. moduleauthor:: Matthias Cuntz

The following functions are provided:

.. autosummary::
   log_bin_spectrum

History
   * Written, Nov 2022, Bernard Heinesch
   * renamed to log_bin_spectrum, Sep 2026, Matthias Cuntz

"""
import math
import warnings
import numpy as np


__all__ = ['log_bin_spectrum']


def log_bin_spectrum(f, y, N, f_min, f_max):
    """
    Log-bin average y(f), from InnFlux code

    Parameters
    ----------
    f : numpy arr of float
        frequency axis [s-1]
    y : numpy arr of float
        Cospectrum
    N : int
        number of bins
    f_min : float
        lower frequency to be considered
    f_max : float
        highest frequency to be considered

    Returns
    -------
    f_out: numpy arr of float, log-binned frequency
    y_out: numpy arr of float, log-binned (co)spectrum

    """
    warnings.filterwarnings("ignore")

    bounds = np.concatenate((np.array([0]),
                             np.exp(np.linspace(math.log(f_min),
                                                math.log(f_max), N)) ),
                            axis=0)

    f_out = np.full([N], np.nan)
    y_out = np.full([N], np.nan)

    f_out[0] = 0  # first bin contains only the f=0 value
    y_out[0] = y[0]

    for i, name in enumerate(f_out[0:N-1], start=1):
        f_out[i] = np.exp( (np.log(bounds[i]) + np.log(bounds[i+1]) )/2 )

        idx = np.logical_and(f > bounds[i], f <= bounds[i+1])
        y_out[i] = np.mean(y[idx])

    return [f_out, y_out]
