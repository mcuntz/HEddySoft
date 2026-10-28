#!/usr/bin/env python
"""
Functions ignoring NaN

This module was written by Bernard Heinesch, University of Liege,
Gembloux Agro-Bio Tech, adapted by Matthias Cuntz while at Institut
National de Recherche pour l'Agriculture, l'Alimentation et
l'Environnement (INRAE), Nancy, France.

:copyright: Copyright 2025- Bernard Heinesch, Matthias Cuntz, see AUTHORS.rst for details.
:license: MIT License, see LICENSE for details.

.. moduleauthor:: Matthias Cuntz

The following functions are provided:

.. autosummary::
   nanlinfit
   nandetrend

History
   * Written, 2025, Bernard Heinesch
   * Merged nanlinfit and nandetrand, Sep 2026, Matthias Cuntz
   * ToDo: t of nanlinfit consider; isnan.any

"""
import numpy as np
from scipy import signal


__all__ = ['nanlinfit', 'nandetrend']


def nanlinfit(y):
    """
    Calculate linear fit for y assuming equidistant x from 0 to N,
    ignoring NaNs
    
    Parameters
    ----------
    y : 1-D ndarray
        Values to fit

    Returns
    -------
    tuple
        slope and offset of linear fit

    """
    y = np.delete(y, np.where(np.isnan(y)))
    #MC: should be done before and indexed as y
    t = np.transpose(np.arange(0, len(y)))
    
    coeff = np.polyfit(t, y, 1)

    slope = coeff[0]
    offset = coeff[1]

    return (slope, offset)


def nandetrend(y):
    """
    Remove linear trend from a vector that may contain NaNs
    
    Parameters
    ----------
    y : 1-D ndarray
        Values to detrend

    Returns
    -------
    1D-array
        Detrended y values

    """
    nanidx = np.where(np.isnan(y))
    if len(nanidx) > 0:
        coeff = nanlinfit(y)
        y = y - np.transpose(np.polyval(coeff, np.arange(0, len(y))))
    else:
        y = signal.detrend(y)

    return y
