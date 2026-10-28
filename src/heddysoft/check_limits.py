#!/usr/bin/env python
"""
Check plausibility limits

This module was written by Matthias Cuntz while at Institut National de Recherche
pour l'Agriculture, l'Alimentation et l'Environnement (INRAE), Nancy, France.

:copyright: Copyright 2026- Matthias Cuntz, see AUTHORS.rst for details.
:license: MIT License, see LICENSE for details.

.. moduleauthor:: Matthias Cuntz

The following functions are provided:

.. autosummary::
   check_limits

History
   * Written, Oct 2025, Matthias Cuntz

"""
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
from .mad import mad


__all__ = ['check_limits']


def check_limits(idata, limits):
    """
    Flag data that is not within limits

    Parameters:
    -----------
    idata : np.ndarray
        1D input data array
    limits: array-like
        (min, max) of possible idata values

    Returns:
    --------
    pandas.DataFrame, numpy.ndarray
        idata with all data set to NaN that is outside limits,
        indexes of flagged data

    """
    odata = idata.copy()

    ii = np.where((idata < limits[0]) | (idata > limits[1]))[0]
    if ii.size > 0:
        odata[ii] = np.nan

    return odata, ii
