#!/usr/bin/env python
"""
Translate Li-7200 error codes

This module was written by Bernard Heinesch, University of Liege,
Gembloux Agro-Bio Tech and modified and simplified by Matthias Cuntz
while at Institut National de Recherche pour l'Agriculture,
l'Alimentation et l'Environnement (INRAE), Nancy, France.

:copyright: Copyright 2025- Bernard Heinesch, Matthias Cuntz, see AUTHORS.rst for details.
:license: MIT License, see LICENSE for details.

.. moduleauthor:: Matthias Cuntz

The following functions are provided:

.. autosummary::
   li7200_diag2df

History
   * Written, 2025, Bernard Heinesch
   * Rename to li7200_diag2df, Sep 2026, Matthias Cuntz

"""
import numpy as np
import pandas as pd


__all__ = ['li7200_diag2df']


# def _int_to_binary_vectorized(arr):
#     """
#     Integer to binary conversion
    
#     Parameters
#     ----------
#     arr : array_like
#         Integer values

#     Returns
#     -------
#     ndarray
#         binary representation of arr

#     """    
#     int2binary = np.vectorize(lambda x: '000' + format(x, 'b'))

#     return int2binary(arr)


def _split_bin_diag_vectorized(diag_val):
    """
    Integer to binary conversion
    
    Parameters
    ----------
    arr : array_like
        Bit representation of flag values

    Returns
    -------
    pandas.DataFrame
        DataFrame with individual diagnostic columns
        for each instrument part:
            'AGC', 'Sync', 'PLL', 'Detector', 'Chopper', 'DeltaPressure',
            'Aux_input', 'Tinlet', 'Toutlet', 'Head detect'

    """    
    # Create the binary array from the input series
    binary_array = np.array([ list(x) for x in diag_val.astype(str) ])
    
    # Create a mask for the AGC columns, i.e. the 4 last bits,
    # hence the last 4 columns of the array
    agc_mask = binary_array[:, -4:].astype(int)
    num_rows, num_cols = agc_mask.shape
    # Use the bitwise left shit operator (<<) to comnbine the binary digits
    # and get one number per row without looping over all the records
    result = np.zeros(num_rows, dtype=int)
    for i in range(num_cols):
        result = result << 1
        result = result | agc_mask[:, i]
        
    # Convert the agc diagnostic into its real value to compare it to the
    # reference value (*6.67 according to the manual)
    agc_column = (result * 6.67).astype(float) 
    # store the mean value
    agc_count = np.mean(agc_column)
    
    # Create a mask for the remaining columns
    other_mask = binary_array[:, 3:-4].astype(int)
    other_count = (~(other_mask==1)).sum(axis=0)
    
    # Create the count dataframe from the counts.
    # First are the agc counts and then the other counts
    # but flipped since the bits must be read  right to left
    # to correspond to the variable names
    diag_count = pd.DataFrame([agc_count] + list(np.flip(other_count)))
    diag_count = diag_count.T
    diag_count.columns = ['AGC', 'Sync', 'PLL', 'Detector', 
                          'Chopper', 'DeltaPressure',
                          'Aux_input', 'Tinlet', 'Toutlet', 
                          'Head detect']
    
    return diag_count


def li7200_diag2df(data):
    """
    Read diagnostic value and translate them to DataFrame
    with individual flag columns for each instrument part
    
    Parameters
    ----------
    flag : array_like
        Time series with binary flag values

    Returns
    -------
    pandas.DataFrame
        DataFrame with individual diagnostic columns
        for each instrument part:
            'AGC', 'Sync', 'PLL', 'Detector', 'Chopper', 'DeltaPressure',
            'Aux_input', 'Tinlet', 'Toutlet', 'Head detect'

    """
    diag_val = data.astype(int)

    # integer to binary conversion
    # diag_val = _int_to_binary_vectorized(diag_val)
    int2binary = np.vectorize(lambda x: format(x, '016b'))
    diag_val = int2binary(diag_val)
    
    diag_count = _split_bin_diag_vectorized(diag_val)

    # print('Number and type of LI7200 diagnostic flags raised:')
    # print(diag_count.loc[:,(diag_count!=0).any()])
   
    return diag_count
