#!/usr/bin/env python
"""
Output DataFrames

This module was written by Matthias Cuntz while at Institut
National de Recherche pour l'Agriculture, l'Alimentation et
l'Environnement (INRAE), Nancy, France.

:copyright: Copyright 2026- Matthias Cuntz, see AUTHORS.rst for details.
:license: MIT License, see LICENSE for details.

.. moduleauthor:: Matthias Cuntz

The following functions are provided:

.. autosummary::
   create_ouput_dataframe
   create_timelag_dataframe

History
   * Written Oct 2026, Matthias Cuntz

"""
import numpy as np
import pandas as pd


__all__ = ['create_ouput_dataframe', 'create_timelag_dataframe']


def create_ouput_dataframe(indx, file_length):
    """
    Create the output data frame

    Parameters
    ----------
    indx : numpy.datetime64
        Datetime for index
    file_length : int
        Length of timestep in seconds

    Returns
    -------
    pandas.DataFrame
        DataFrame with all entries NaN

    """
    odf = pd.DataFrame({}, index=[indx])
    odf.index.name = 'TIMESTAMP_START'

    odf['TIMESTAMP_END'] = indx + np.timedelta64(file_length, 's')

    odf['air_pressure (Pa)'] = np.nan
    odf['air_relative_humidity (0-1)'] = np.nan
    odf['air_sat (Pa)'] = np.nan
    odf['air_temperature (K)'] = np.nan
    odf['air_vpd (Pa)'] = np.nan
    odf['complete_sonic_ts (0-1)'] = np.nan
    odf['complete_sonic_wind (0-1)'] = np.nan
    odf['ea (Pa)'] = np.nan
    odf['ea_dry (Pa)'] = np.nan
    odf['esat (Pa)'] = np.nan
    odf['H (W m-2)'] = np.nan
    odf['H_qc'] = 2
    odf['itc_T (%)'] = np.nan
    odf['itc_u (%)'] = np.nan
    odf['itc_w (%)'] = np.nan
    odf['L (m)'] = np.nan
    odf['lambda (J kg-1)'] = np.nan
    odf['num_spikes_ts (#)'] = np.nan
    odf['num_spikes_w (#)'] = np.nan
    odf['rho_air (mol m-3)'] = np.nan
    odf['rho_air_dry (mol m-3)'] = np.nan
    odf['rho_cp (J K-1 m-3)'] = np.nan
    odf['sigma_w (m s-1)'] = np.nan
    odf['steady_state_wT_FokenWichura1996 (%)'] = np.nan
    odf['u_star (m s-1)'] = np.nan
    odf['uu_mean (m2 s-2)'] = np.nan
    odf['uw_mean (m2 s-2)'] = np.nan
    odf['V_mol (m3 mol-1)'] = np.nan
    odf['V_mol_dry (m3 mol-1)'] = np.nan
    odf['vv_mean (m2 s-2)'] = np.nan
    odf['vw_mean (m2 s-2)'] = np.nan
    odf['wind_direction_mean (degree)'] = np.nan
    odf['wind_direction_std (degree)'] = np.nan
    odf['wind_speed_mean (m s-1)'] = np.nan
    odf['wind_speed_std (m s-1)'] = np.nan
    odf['wT (K m s-1)'] = np.nan
    odf['ww_mean (m2 s-2)'] = np.nan
    odf['z/L'] = np.nan
    
    for TT in ['CO2', 'H2O']:
        tt = TT.lower()
        if tt == 'co2':
            # CO2 is in umol/mol
            mol = 'umol'
        elif tt == 'h2o':
            # H2O is in mmol/mol
            mol = 'mmol'
        else:
            raise ValueError(f'Gas not known: {TT}')
        odf[f'complete_irga_{tt} (0-1)'] = np.nan
        odf[f'F_{tt} ({mol} m-2 s-1)'] = np.nan
        odf[f'F_{tt}_noise_mean ({mol} m-2 s-1)'] = np.nan
        odf[f'F_{tt}_noise_rmse ({mol} m-2 s-1)'] = np.nan
        odf[f'F_{tt}_noise_std ({mol} m-2 s-1)'] = np.nan
        odf[f'F_{tt}_qc'] = 2
        odf[f'F_{tt}_random_error_FinkelsteinSims2001 ({mol} m-2 s-1)'] = np.nan
        odf[f'F_{tt}_random_error_noise ({mol} m-2 s-1)'] = np.nan
        odf[f'F_{tt}_random_flux ({mol} m-2 s-1)'] = np.nan
        odf[f'lag_samples_{tt} (#)'] = np.nan
        odf[f'lag_time_{tt} (s)'] = np.nan
        odf[f'num_limits_{tt} (#)'] = np.nan
        odf[f'num_spikes_{tt} (#)'] = np.nan
        odf[f'steady_state_wc_{tt}_FokenWichura1996 (%)'] = np.nan
        odf[f'wc_{tt} ({mol} mol-1 m s-1)'] = np.nan

    odf['E (kg m-2 s-1)'] = np.nan
    odf['E_qc'] = 2
    odf['LE (W m-2)'] = np.nan
    odf['LE_qc'] = 2
    
    return odf


def create_spectral_dictionary(indx, file_length, num_freq_bins):
    """
    Create the spectral output dictionary

    Parameters
    ----------
    indx : numpy.datetime64
        Datetime for index
    file_length : int
        Length of timestep in seconds

    Returns
    -------
    dict
        Dictionary with three DataFrames with all entries NaN

    """
    # scalar values
    idf = pd.DataFrame({}, index=[indx])
    idf.index.name = 'TIMESTAMP_START'

    idf['TIMESTAMP_END'] = indx + np.timedelta64(file_length, 's')

    idf['wT'] = np.nan
    idf['wind_speed'] = np.nan
    idf['wind_direction'] = np.nan
    idf['u_star'] = np.nan
    idf['z/L'] = np.nan
    idf['num_spikes_w'] = np.nan
    idf['itc_w'] = np.nan
    idf['steady_state_wc_co2_Mahrt1998'] = np.nan
    idf['steady_state_wc_h2o_Mahrt1998'] = np.nan
    idf['F_co2'] = np.nan
    idf['F_h2o'] = np.nan

    # len 12 variables
    pdf = pd.DataFrame()

    pdf['ipt_w'] = np.full(12, np.nan)
    pdf['ipt_co2'] = np.full(12, np.nan)
    pdf['ipt_h2o'] = np.full(12, np.nan)

    # len num_freq_bins (50) variables
    sdf = pd.DataFrame()

    sdf['freq'] = np.full(num_freq_bins, np.nan)
    sdf['cospec_wT_scaled'] = np.full(num_freq_bins, np.nan)
    sdf['spec_co2_scaled'] = np.full(num_freq_bins, np.nan)
    sdf['cospec_wc_co2_scaled'] = np.full(num_freq_bins, np.nan)
    sdf['spec_h2o_scaled'] = np.full(num_freq_bins, np.nan)
    sdf['cospec_wc_h2o_scaled'] = np.full(num_freq_bins, np.nan)

    out = {'scalar': idf, 'instrument': pdf, 'spectral': sdf}

    return out


def create_timelag_dataframe(indx, file_length):
    """
    Create the timelag output data frame

    Parameters
    ----------
    indx : numpy.datetime64
        Datetime for index
    file_length : int
        Length of timestep in seconds

    Returns
    -------
    pandas.DataFrame
        DataFrame with all entries NaN

    """
    odf = pd.DataFrame({}, index=[indx])
    odf.index.name = 'TIMESTAMP_START'

    odf['TIMESTAMP_END'] = indx + np.timedelta64(file_length, 's')

    odf['wind_speed_mean (m s-1)'] = np.nan
    odf['wind_direction_mean (degree)'] = np.nan
    odf['u_star (m s-1)'] = np.nan
    odf['sigma_w (m s-1)'] = np.nan
    odf['air_pressure (Pa)'] = np.nan
    odf['air_temperature (K)'] = np.nan
    odf['air_relative_humidity (0-1)'] = np.nan
    odf['air_vpd (Pa)'] = np.nan
    odf['L (m)'] = np.nan
    odf['z/L'] = np.nan
    for TT in ['CO2', 'H2O']:
        tt = TT.lower()
        odf[f'lag_samples_{tt} (#)'] = np.nan
        odf[f'lag_time_{tt} (s)'] = np.nan
    
    return odf
