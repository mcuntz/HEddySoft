#!/usr/bin/env python
"""
Only timelags of eddy covariance data

This module was written by Matthias Cuntz while at Institut National de
Recherche pour l'Agriculture, l'Alimentation et l'Environnement (INRAE),
Nancy, France.

:copyright: Copyright 2026- Matthias Cuntz, see AUTHORS.rst for details.
:license: MIT License, see LICENSE for details.

.. moduleauthor:: Matthias Cuntz

The following functions are provided:

.. autosummary::
   heddysoft_timelag

History
    * Written, Oct 2026 from heddysoft_main.py

"""
import os
import numpy as np
from .air import g, T0, vonKarman
from .air import esat
from .config import read_config, get_config_base_dir
from .create_ouput_dataframes import create_timelag_dataframe
from .lagged_covariance import cov, lagged_covariance
from .li7200_diag2df import li7200_diag2df
from .nanfunctions import nandetrend
from .read_files import read_raw_datafile
from .spike_detection import spike_detection_vickers_mahrt_1997
from .wind_direction import wind_direction
from .wind_rotation import wind_rotation


__all__ = ['heddysoft_timelag']


def heddysoft_timelag(config, df_meteo, output_file, ifile,
                      timelag_dict={}, rotation_matrix=None):
    """
    Process a single sonic/irga file for timelags
    
    Parameters
    ----------
    config : str or astConfigParser
        Config (filename)
    df_meteo : pandas.DataFrame
        DataFrame with temperature, pressure, and relative humidity
    output_file : str
        Name of outputfile include including path
    ifile : str
        Filename with absolute path
    timelag_dict : dict
        Dictionary with appropriate information for timelag detection
    rotation_matrix: dict of 3x3 nd-arrays, optional
        Needed only for angle-dependent planar fit

    Returns
    -------
    None

    """
    print(f'    Process {ifile}')

    cfg = read_config(config)

    # plots start with this string
    base_dir = get_config_base_dir(cfg)
    plot_path = cfg.get('plot', 'dir')
    if not (plot_path.startswith('/') or (plot_path[1] == ':')):
        plot_path = f'{base_dir}/{plot_path}'
    plot_base = plot_path + '/' + os.path.basename(ifile)

    # read sonic and irga file
    df = read_raw_datafile(ifile, cfg)

    # start output DataFrame; need [] around index
    file_length = cfg.get('sonic', 'file_length')
    odf = create_timelag_dataframe(df.index[0].replace(microsecond=0),
                                   file_length)

    # set sonic and irga flags
    sonic_names = [ s.strip() for s in cfg.get('sonic', 'names').split(',') ]
    if 'sonic_flag' in sonic_names:
        sonic_columns = list(cfg.get('sonic', 'columns'))
        sonic_columns_wo_flag = sonic_columns.copy()
        sonic_names_wo_flag = sonic_names.copy()
        findx = sonic_names.index('sonic_flag')
        _ = sonic_names_wo_flag.pop(findx)
        _ = sonic_columns_wo_flag.pop(findx)
        if any(df['sonic_flag'] != 0):
            df.loc[df['sonic_flag'] != 0, sonic_names_wo_flag] = np.nan

    irga_names = [ s.strip() for s in cfg.get('irga', 'names').split(',') ]
    if 'irga_flag' in irga_names:
        irga_columns = list(cfg.get('irga', 'columns'))
        irga_columns_wo_flag = irga_columns.copy()
        irga_names_wo_flag = irga_names.copy()
        findx = irga_names.index('irga_flag')
        _ = irga_names_wo_flag.pop(findx)
        _ = irga_columns_wo_flag.pop(findx)
        diag_count = li7200_diag2df(df['irga_flag'])
        if (diag_count.drop(
                columns=['AGC', 'Sync', 'Aux_input']) != 0).any(
                    axis=1).any():
            print(f'All irga data flagged in {ifile}.')
            return odf

    #
    # sonic data
    #

    # w, ts spike detection
    sampling_rate = cfg.get('sonic', 'sampling_rate')
    spike_mode = cfg.get('spikes', 'mode')
    if spike_mode != 0:
        spike_max_consec = cfg.get('spikes', 'max_consecutive_spikes')
        plot_spike_detection = cfg.get('plot', 'spike_detection')
        plot_spike = plot_base if plot_spike_detection else ''
        for ss in ['w', 'ts']:
            spike_threshold = cfg.get('spikes', f'threshold_{ss}')
            sdf, is_spike, nspikes = spike_detection_vickers_mahrt_1997(
                df[ss].to_numpy(),
                spike_mode=spike_mode,
                avrg_len=int(file_length / 60),
                ac_freq=sampling_rate,
                spike_limit=spike_threshold,
                max_consec_spikes=spike_max_consec,
                error_value=np.nan,
                series_label=ss,
                ctrplot=plot_spike)
            df[ss] = sdf

    # Coordinate rotations
    # mean and standard deviation of horizontal wind speed and direction
    # note: wind direction is the angle where the wind is coming from,
    #       with north = 0 and east = 90 degrees
    mdf = df[['u', 'v', 'w']].mean(axis=0)
    # windspeed
    ws_mean = np.sqrt(mdf['u']**2 + mdf['v']**2 + mdf['w']**2)
    # wind direction
    sonic_type = cfg.get('sonic', 'sonic_type')
    sonic_orientation = cfg.get('sonic', 'orientation')
    wd_mean = wind_direction(
        mdf['u'], mdf['v'], sonic_type, sonic_orientation)
    # coordinate rotations
    rotation_mode = cfg.get('rotation', 'mode')
    uvw_rot, rot_phi, R_tilt = wind_rotation(
        df, rotation_mode=rotation_mode, wd_mean=wd_mean,
        rotation_matrix=rotation_matrix)
    # uvw_rot_mean = np.nanmean(uvw_rot, axis=0)
    df['u_rot'] = uvw_rot[:, 0]
    df['v_rot'] = uvw_rot[:, 1]
    df['w_rot'] = uvw_rot[:, 2]

    odf['wind_speed_mean (m s-1)'] = ws_mean
    odf['wind_direction_mean (degree)'] = wd_mean

    # interpolate wind and scalar data
    df['u_rot_f'] = df['u_rot'].interpolate(method='nearest')
    df['v_rot_f'] = df['v_rot'].interpolate(method='nearest')
    df['w_rot_f'] = df['w_rot'].interpolate(method='nearest')
    df['ts_f'] = df['ts'].interpolate(method='nearest')
    df['ts_f'] += T0

    # Reynolds decomposition of wind components (e.g. u = u_mean + u_prime)
    df['u_prime'] = df['u_rot_f'] - df['u_rot_f'].mean()
    df['v_prime'] = df['v_rot_f'] - df['v_rot_f'].mean()
    df['w_prime'] = df['w_rot_f'] - df['w_rot_f'].mean()
    # check for Nans and replace NaN by zero
    if df['u_prime'].isna().any():
        print("      Warning: NaN values detected in u': replacing with 0.")
        df['u_prime'] = df['u_prime'].fillna(0.)
    if df['v_prime'].isna().any():
        print("      Warning: NaN values detected in v': replacing with 0.")
        df['v_prime'] = df['v_prime'].fillna(0.)
    if df['w_prime'].isna().any():
        print("      Warning: NaN values detected in w': replacing with 0.")
        df['w_prime'] = df['w_prime'].fillna(0.)

    # covariances of wind components
    # uv = (df['u_prime'] * df['v_prime']).mean()
    uw = (df['u_prime'] * df['w_prime']).mean()
    vw = (df['v_prime'] * df['w_prime']).mean()

    # friction velocity, u*
    u_star = (uw**2 + vw**2)**0.25

    odf['u_star (m s-1)'] = u_star
    odf['sigma_w (m s-1)'] = df['w_prime'].std(ddof=1)

    # temperature T from virtual sonic temperature T_sonic
    # and water vapor concentration
    #   T_sonic = T * (1 + 0.32 * w)
    # were w is the volume mixing ratio of water vapor in air
    detrend_T = cfg.get('param', 'detrend_temperature_signal')
    df['h2o_f'] = df['H2O_dry'].interpolate(method='nearest')
    # H2O data is in mmol/mol, i.e. ea/p * 1000.
    df['T'] = df['ts_f'] / (1. + 0.32 * 1.e-3 * df['h2o_f'])
    if detrend_T:
        df['T_prime'] = nandetrend(df['T'])
    else:
        df['T_prime'] = df['T'] - df['T'].mean()
    if df['T_prime'].isna().any():
        print("      Warning: NaN values detected in T': replacing with 0.")
        df['T_prime'] = df['T_prime'].fillna(0.)

    # mean meteo
    # assume that time of meteo data is the end of the half-hour period
    # i.e. close to df.index[-1]
    idx = np.argmin(np.abs(df_meteo.index - df.index[-1]))
    idx = df_meteo.index[idx]
    ipress = cfg.get('meteo', 'pressure')
    itemp = cfg.get('meteo', 'temperature')
    irh = cfg.get('meteo', 'relative_humidity')
    # # Bernard meteo data
    # p_meteo = df_meteo.loc[idx, ipress] * 100.  # Pa
    # T_meteo = df_meteo.loc[idx, itemp] + T0     # K
    # FR-Hes biomet data
    p_meteo = df_meteo.loc[idx, ipress]       # Pa
    T_meteo = df_meteo.loc[idx, itemp]        # K
    rh_meteo = df_meteo.loc[idx, irh] / 100.  # 0-1

    # calculate temperature flux <w'T'>
    wT = cov(df, 'w_prime', 'T_prime')

    odf['air_pressure (Pa)'] = p_meteo
    odf['air_temperature (K)'] = T_meteo
    odf['air_relative_humidity (0-1)'] = rh_meteo
    odf['air_vpd (Pa)'] = esat(T_meteo) * (1. - rh_meteo)

    # Obukhov length, L
    L = -T_meteo * u_star**3 / (vonKarman * g * wT)

    # stability parameter, z/L
    sonic_height = cfg.get('sonic', 'sensor_height')
    zoL = sonic_height / L

    odf['L (m)'] = L
    odf['z/L'] = zoL

    #
    # irga data
    #

    detrend_irga = cfg.get('param', 'detrend_irga_signal')
    for TT in ['CO2', 'H2O']:
        Tname = f'{TT}_dry'
        tt = TT.lower()
        # spike detection
        if spike_mode != 0:
            spike_threshold = cfg.get('spikes', 'threshold_irga')
            sdf, is_spike, nspikes = spike_detection_vickers_mahrt_1997(
                df[Tname].to_numpy(),
                spike_mode=spike_mode,
                avrg_len=int(file_length / 60),
                ac_freq=sampling_rate,
                spike_limit=spike_threshold,
                max_consec_spikes=spike_max_consec,
                error_value=np.nan,
                series_label=Tname,
                ctrplot=plot_spike)
            df[Tname] = sdf

        # interpolate
        df[f'{tt}_f'] = df[Tname].interpolate(method='nearest')

        # Reynolds decomposition
        if detrend_irga:
            df[f'{tt}_prime'] = nandetrend(df[f'{tt}_f'])
        else:
            df[f'{tt}_prime'] = df[f'{tt}_f'] - df[f'{tt}_f'].mean()
        if df[f'{tt}_prime'].isna().any():
            print(f'NaN in {TT} detected',
                  df[f'{tt}_prime'].loc[df[f'{tt}_prime'].isna()])
            continue

        # covariances and lag times
        plot_covariance_peak = cfg.get('plot', 'covariance_peak')
        plot_covariance = (plot_base, tt) if plot_covariance_peak else tuple()
        wc, lag_samples = lagged_covariance(
            tt, cfg, w_prime=df['w_prime'], c_prime=df[f'{tt}_prime'],
            timelag_dict=timelag_dict, rh=rh_meteo,
            plot_label=plot_covariance)

        odf[f'lag_samples_{tt} (#)'] = lag_samples
        odf[f'lag_time_{tt} (s)'] = lag_samples / sampling_rate

    # append to output file
    if os.path.exists(output_file):
        header = False
    else:
        header = True
    odf.to_csv(output_file, na_rep=-9999, mode='a', header=header)

    return
