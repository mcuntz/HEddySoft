#!/usr/bin/env python
"""
Process eddy covariance data

This module was written by Bernard Heinesch, University of Liege,
Gembloux Agro-Bio Tech and modified and simplified by Matthias Cuntz
while at Institut National de Recherche pour l'Agriculture,
l'Alimentation et l'Environnement (INRAE), Nancy, France.

:copyright: Copyright 2025- Bernard Heinesch, Matthias Cuntz, see AUTHORS.rst for details.
:license: MIT License, see LICENSE for details.

.. moduleauthor:: Matthias Cuntz

The following functions are provided:

.. autosummary::
   heddysoft_main

History
    * Written, Sep 2026 from GEddySoft_main.py by Bernard Heinesch

"""
import os
import pickle
import numpy as np
from .air import g, Mv, Rgas, T0, vonKarman
from .air import esat, lambda_evap, rhocp
from .check_limits import check_limits
from .config import read_config, get_config_base_dir
from .cospectrum import cospectrum
from .create_ouput_dataframes import create_ouput_dataframe
from .create_ouput_dataframes import create_spectral_dictionary
from .flux_uncertainties import flux_uncertainties
from .get_files import get_dates_from_files
from .instrument_problems_tests import instrument_problems_tests
from .lagged_covariance import cov, lagged_covariance
from .li7200_diag2df import li7200_diag2df
from .log_bin_spectrum import log_bin_spectrum
from .nanfunctions import nandetrend
from .read_files import read_raw_datafile
from .spike_detection import spike_detection_vickers_mahrt_1997
from .spike_detection import spike_detection_mauder_et_al_2013
from .test_itc import test_itc
from .test_steady_state import test_steady_state_FW96, test_steady_state_M98
from .wet2dry_mixing_ratio import wet2dry_mixing_ratio
from .wind_direction import wind_direction
from .wind_rotation import wind_rotation


__all__ = ['heddysoft_main']


def _save_df(df, ofile):
    """
    Append DataFrame to file

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to save
    ofile : str
        Name of outputfile include including path

    Returns
    -------
    None

    """
    # append to output file
    if os.path.exists(ofile):
        header = False
    else:
        header = True
    df.to_csv(ofile, na_rep=-9999, mode='a', header=header)

    return


def heddysoft_main(config, df_meteo, output_file, ifile,
                   timelag_dict={}, rotation_matrix=None):
    """
    Process a single sonic/irga file

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
    odf = create_ouput_dataframe(df.index[0].replace(microsecond=0),
                                 file_length)

    write_spectral_files = cfg.get('param', 'write_spectral_files')
    num_freq_bins = cfg.get('param', 'num_freq_bins')
    if write_spectral_files:
        sdf = create_spectral_dictionary(df.index[0].replace(microsecond=0),
                                         file_length, num_freq_bins)
        scalar_df = sdf['scalar']
        instrument_df = sdf['instrument']
        spectral_df = sdf['spectral']

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

    irga_type = cfg.get('irga', 'irga_type')
    irga_names = [ s.strip() for s in cfg.get('irga', 'names').split(',') ]
    if irga_type.lower().startswith('li7200') and ('irga_flag' in irga_names):
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
            print(f'        IRGA data flagged in {ifile}.')
            _save_df(odf, output_file)
            return
        # print('      Number and type of LI7200 diagnostic flags raised:')
        # print(diag_count.loc[:, (diag_count != 0).any()])

    #
    # Convert to dry mixing ratio if needed
    #

    # assures also that CO2_dry and H2O_dry are present
    if not irga_type.lower().startswith('li7200'):
        df = wet2dry_mixing_ratio(df, cfg)
    else:
        co2_name = ''
        h2o_name = ''
        for ii in irga_names:
            if ii.lower().startswith('co2'):
                co2_name = ii
            if ii.lower().startswith('h2o'):
                h2o_name = ii
        if ((co2_name == '') or (h2o_name == '')):
            raise ValueError(f'irga names must include names starting with CO2'
                             f' and H2O; current names: {irga_names}')
        df['CO2_dry'] = df[co2_name]
        df['H2O_dry'] = df[h2o_name]

    #
    # Plausibility limits
    #

    dlim = {'u': cfg.get('sonic', 'u_lim'),
            'v': cfg.get('sonic', 'v_lim'),
            'w': cfg.get('sonic', 'w_lim'),
            'ts': cfg.get('sonic', 'ts_lim')}
    for ss in ['u', 'v', 'w', 'ts']:
        ldf, ii = check_limits(df[ss].to_numpy(), dlim[ss])
        df[ss] = ldf
        odf[f'num_limits_{ss} (#)'] = ii.size

    dlim = {'CO2_dry': cfg.get('irga', 'co2_lim'),
            'H2O_dry': cfg.get('irga', 'h2o_lim')}
    for TT in ['CO2', 'H2O']:
        Tname = f'{TT}_dry'
        tt = TT.lower()

        ldf, ii = check_limits(df[Tname].to_numpy(), dlim[Tname])
        df[Tname] = ldf
        odf[f'num_limits_{ss} (#)'] = ii.size

    #
    # 1st check that enough data to catch faulty files
    #

    sampling_rate = cfg.get('sonic', 'sampling_rate')
    nsoll = file_length * sampling_rate
    nsoll_ratio = cfg.get('param', 'data_complete')
    nist = df[sonic_names[:3]].notna().sum(axis=0)  # u, v, w
    ratio_u = nist.min() / nsoll
    odf['complete_sonic_wind (0-1)'] = ratio_u
    if ratio_u < nsoll_ratio:
        print(f'        Not enough sonic data in {ifile}, only about'
              f' {ratio_u * 100:.2f}% (config param.data_complete:'
              f' {nsoll_ratio * 100:.2f}%).')
        _save_df(odf, output_file)
        return

    for TT in ['CO2', 'H2O']:
        Tname = f'{TT}_dry'
        tt = TT.lower()

        ratio_irga = df[Tname].notna().sum() / nsoll
        odf[f'complete_irga_{tt} (0-1)'] = ratio_irga
        if ratio_irga < nsoll_ratio:
            print(f'        Not enough {TT} data after high frequency checks'
                  f' in {ifile}, only about {ratio_irga * 100:.2f}% (config'
                  f' param.data_complete: {nsoll_ratio * 100:.2f}%).')
            _save_df(odf, output_file)
            return

    #
    # Spike detection
    #

    # w, ts spike detection
    spike_mode = cfg.get('spikes', 'mode')
    if spike_mode != 0:
        plot_spike_detection = cfg.get('plot', 'spike_detection')
        plot_spike = plot_base if plot_spike_detection else ''
        spike_method = cfg.get('spikes', 'method')
        if spike_method.lower().startswith('vickers'):
            spike_max_consec = cfg.get('spikes', 'max_consecutive_spikes')
            for ss in ['w', 'ts']:
                spike_threshold = cfg.get('spikes', f'threshold_{ss}')
                idf, ii_spike, nspikeevents = spike_detection_vickers_mahrt_1997(
                    df[ss].to_numpy(),
                    spike_mode=spike_mode,
                    avrg_len=int(file_length / 60),
                    ac_freq=sampling_rate,
                    spike_limit=spike_threshold,
                    max_consec_spikes=spike_max_consec,
                    error_value=np.nan,
                    series_label=ss,
                    ctrplot=plot_spike)
                df[ss] = idf
                if ss == 'w':
                    num_spikes_w = int(sum(ii_spike))
                else:
                    num_spikes_ts = int(sum(ii_spike))
        elif spike_method.lower().startswith('mauder'):
            z_mad = cfg.get('spikes', 'z_mad')
            for ss in ['w', 'ts']:
                idf, ii_spike = spike_detection_mauder_et_al_2013(
                    df[ss].to_numpy(), z=z_mad,
                    spike_mode=spike_mode,
                    plot=plot_spike, plot_label=ss)
                df[ss] = idf
                if ss == 'w':
                    num_spikes_w = int(sum(ii_spike))
                else:
                    num_spikes_ts = int(sum(ii_spike))
        else:
            raise ValueError(f'Spike method not known: {spike_method}')
    else:
        num_spikes_w = np.nan
        num_spikes_ts = np.nan

    odf['num_spikes_w (#)'] = num_spikes_w
    odf['num_spikes_ts (#)'] = num_spikes_ts

    if write_spectral_files:
        scalar_df['num_spikes_w'] = num_spikes_w

    # irga spike detection
    for TT in ['CO2', 'H2O']:
        Tname = f'{TT}_dry'
        tt = TT.lower()
        # spike detection
        if spike_mode != 0:
            if spike_method.lower().startswith('vickers'):
                spike_threshold = cfg.get('spikes', 'threshold_irga')
                idf, ii_spike, nspikeevents = spike_detection_vickers_mahrt_1997(
                    df[Tname].to_numpy(),
                    spike_mode=spike_mode,
                    avrg_len=int(file_length / 60),
                    ac_freq=sampling_rate,
                    spike_limit=spike_threshold,
                    max_consec_spikes=spike_max_consec,
                    error_value=np.nan,
                    series_label=Tname,
                    ctrplot=plot_spike)
                df[Tname] = idf
                num_spikes = int(sum(ii_spike))
            elif spike_method.lower().startswith('mauder'):
                idf, ii_spike = spike_detection_mauder_et_al_2013(
                    df[Tname].to_numpy(), z=z_mad,
                    spike_mode=spike_mode,
                    plot=plot_spike, plot_label=Tname)
                df[Tname] = idf
                num_spikes = sum(ii_spike)
        else:
            num_spikes = np.nan

        odf[f'num_spikes_{tt} (#)'] = num_spikes

    #
    # Instrument tests
    #

    # tests on instrument malfunctions from Vitale et al. (2020)
    detrend_T = cfg.get('param', 'detrend_temperature_signal')
    detrend_irga = cfg.get('param', 'detrend_irga_signal')
    if write_spectral_files:
        plot_instrument_problems_tests = cfg.get('plot',
                                                 'instrument_problems_tests')
        plot_problems_tests = (plot_base if plot_instrument_problems_tests
                               else '')
        ipt_w = instrument_problems_tests(
            df['w'].to_numpy(), False, sampling_rate, plot_problems_tests, 'w')
        ipt_ts = instrument_problems_tests(
            df['ts'].to_numpy(), detrend_T, sampling_rate,
            plot_problems_tests, 'ts')

        instrument_df['ipt_w'] = ipt_w

        for TT in ['CO2', 'H2O']:
            Tname = f'{TT}_dry'
            tt = TT.lower()

            # tests on instrument malfunctions from Vitale et al. (2020)
            ipt = instrument_problems_tests(
                df[Tname].to_numpy(), detrend_irga, sampling_rate,
                plot_problems_tests, TT)

        instrument_df[f'ipt_{tt}'] = ipt

    #
    # Check enough data after high frequency checks
    #

    # check that enough data
    nist = df[sonic_names[:3]].notna().sum(axis=0)  # u, v, w
    ratio_u = nist.min() / nsoll
    odf['complete_sonic_wind (0-1)'] = ratio_u
    if ratio_u < nsoll_ratio:
        print(f'        Not enough sonic data in {ifile}, only about'
              f' {ratio_u * 100:.2f}% (config param.data_complete:'
              f' {nsoll_ratio * 100:.2f}%).')
        _save_df(odf, output_file)
        return

    for TT in ['CO2', 'H2O']:
        Tname = f'{TT}_dry'
        tt = TT.lower()

        ratio_irga = df[Tname].notna().sum() / nsoll
        odf[f'complete_irga_{tt} (0-1)'] = ratio_irga
        if ratio_irga < nsoll_ratio:
            print(f'        Not enough {TT} data after high frequency checks'
                  f' in {ifile}, only about {ratio_irga * 100:.2f}% (config'
                  f' param.data_complete: {nsoll_ratio * 100:.2f}%).')
            _save_df(odf, output_file)
            return

    #
    # Sonic data
    #

    #
    # Coordinate rotation
    #

    # mean and standard deviation of horizontal wind speed and direction
    # note: wind direction is the angle where the wind is coming from,
    #       with north = 0 and east = 90 degrees
    mdf = df[['u', 'v', 'w']].mean(axis=0)
    # windspeed
    ws_mean = np.sqrt(mdf['u']**2 + mdf['v']**2 + mdf['w']**2)
    ws = np.sqrt(df['u']**2 + df['v']**2 + df['w']**2)
    ws_std = np.std(ws, ddof=1)
    # wind direction
    sonic_type = cfg.get('sonic', 'sonic_type')
    sonic_orientation = cfg.get('sonic', 'orientation')
    wd_mean = wind_direction(
        mdf['u'], mdf['v'], sonic_type, sonic_orientation)
    wd = wind_direction(
        df['u'], df['v'], sonic_type, sonic_orientation)
    wd_std = np.std(wd, ddof=1)
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
    odf['wind_speed_std (m s-1)'] = ws_std
    odf['wind_direction_mean (degree)'] = wd_mean
    odf['wind_direction_std (degree)'] = wd_std

    if write_spectral_files:
        scalar_df['wind_speed'] = ws_mean
        scalar_df['wind_direction'] = wd_mean

    # interpolate wind and scalar data
    df['u_rot_f'] = df['u_rot'].interpolate(method='nearest')
    df['v_rot_f'] = df['v_rot'].interpolate(method='nearest')
    df['w_rot_f'] = df['w_rot'].interpolate(method='nearest')
    df['ts_f'] = df['ts'].interpolate(method='nearest')
    df['ts_f'] += T0

    ratio_ts = df['ts_f'].notna().sum() / nsoll
    odf['complete_sonic_ts (0-1)'] = ratio_ts
    if ratio_ts < nsoll_ratio:
        print(f'        Not enough T_sonic data after interpolation in'
              f' {ifile}, only about {ratio_ts * 100:.2f}%'
              f' (config param.data_complete: {nsoll_ratio * 100:.2f}%).')
        _save_df(odf, output_file)
        return

    # Reynolds decomposition of wind components (e.g. u = u_mean + u_prime)
    df['u_prime'] = df['u_rot_f'] - df['u_rot_f'].mean()
    df['v_prime'] = df['v_rot_f'] - df['v_rot_f'].mean()
    df['w_prime'] = df['w_rot_f'] - df['w_rot_f'].mean()
    # check for Nans and replace NaN by zero
    if df['u_prime'].isna().any():
        print("          Warning: NaN values detected in u': replacing with"
              " 0.")
        df['u_prime'] = df['u_prime'].fillna(0.)
    if df['v_prime'].isna().any():
        print("          Warning: NaN values detected in v': replacing with"
              " 0.")
        df['v_prime'] = df['v_prime'].fillna(0.)
    if df['w_prime'].isna().any():
        print("          Warning: NaN values detected in w': replacing with"
              " 0.")
        df['w_prime'] = df['w_prime'].fillna(0.)

    # covariances of wind components
    # uv = (df['u_prime'] * df['v_prime']).mean()
    uw = (df['u_prime'] * df['w_prime']).mean()
    vw = (df['v_prime'] * df['w_prime']).mean()
    uu = (df['u_prime'] * df['u_prime']).mean()
    vv = (df['v_prime'] * df['v_prime']).mean()
    ww = (df['w_prime'] * df['w_prime']).mean()

    # friction velocity, u*
    u_star = (uw**2 + vw**2)**0.25

    odf['uw_mean (m2 s-2)'] = uw
    odf['vw_mean (m2 s-2)'] = vw
    odf['uu_mean (m2 s-2)'] = uu
    odf['vv_mean (m2 s-2)'] = vv
    odf['ww_mean (m2 s-2)'] = ww
    odf['u_star (m s-1)'] = u_star
    odf['sigma_w (m s-1)'] = df['w_prime'].std(ddof=1)

    if write_spectral_files:
        scalar_df['u_star'] = u_star

    # ToDo? IRGA treatment should be here

    # temperature T from virtual sonic temperature T_sonic
    # and water vapor concentration
    #   T_sonic = T * (1 + 0.32 * w)
    # were w is the volume mixing ratio of water vapor in air
    df['h2o_f'] = df['H2O_dry'].interpolate(method='nearest')
    ratio_h2o = df['h2o_f'].notna().sum() / nsoll
    if ratio_h2o < nsoll_ratio:
        odf['complete_irga_h2o (0-1)'] = ratio_h2o
        print(f'        Not enough H2O data after interpolation in {ifile},'
              f' only about {ratio_h2o * 100:.2f}% (config'
              f' param.data_complete: {nsoll_ratio * 100:.2f}%).')
        _save_df(odf, output_file)
        return
    # H2O data is in mmol/mol, i.e. ea/p * 1000.
    df['T'] = df['ts_f'] / (1. + 0.32 * 1.e-3 * df['h2o_f'])
    if detrend_T:
        df['T_prime'] = nandetrend(df['T'])
    else:
        df['T_prime'] = df['T'] - df['T'].mean()
    if df['T_prime'].isna().any():
        print("          Warning: NaN values detected in T': replacing with"
              " 0.")
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

    # air molar concentration (mol m-3)
    # # molar volume of 1 mole at standard temperature and pressure:
    # #   22.414 l/mol = 22.414e-3 m**3/mol
    # air_mol_conc = 1. / (22.414e-3 * T_meteo / 273.15 * 101325. / p_meteo)
    #
    # Calculations from Sabatini et al. (Int Agrophys 2018)
    # c = p / RT  is the inverse of air molar volume Va = RT/p
    Va = Rgas * T_meteo / p_meteo  # m3 mol-1
    air_mol_conc = 1. / Va         # mol m-3
    # Saturated water vapour partial pressure [Pa]
    es = esat(T_meteo)
    # Ambient water vapour partial pressure [Pa]
    ea = rh_meteo * es
    # Dry air partial pressure [Pa], as P minus water vapour partial pressure
    ed =  p_meteo - ea
    # Dry air molar volume Vd = RT/pd
    Vd = Rgas * T_meteo / ed
    # Specific heat of evaporation [J kg-1 K-1]
    lmbda = lambda_evap(T_meteo)

    # calculate temperature flux <w'T'>
    wT = cov(df, 'w_prime', 'T_prime')

    # rho_air * Cp
    rho_cp = rhocp(T_meteo, rh_meteo, p_meteo)

    odf['air_pressure (Pa)'] = p_meteo
    odf['air_temperature (K)'] = T_meteo
    odf['air_relative_humidity (0-1)'] = rh_meteo
    odf['air_sat (Pa)'] = es
    odf['air_vpd (Pa)'] = es * (1. - rh_meteo)
    odf['wT (K m s-1)'] = wT
    odf['rho_cp (J K-1 m-3)'] = rho_cp
    odf['H (W m-2)'] = rho_cp * wT
    odf['rho_air (mol m-3)'] = air_mol_conc
    odf['V_mol (m3 mol-1)'] = Va
    odf['esat (Pa)'] = es
    odf['ea (Pa)'] = ea
    odf['ea_dry (Pa)'] = ed
    odf['rho_air_dry (mol m-3)'] = 1. / Vd
    odf['V_mol_dry (m3 mol-1)'] = Vd
    odf['lambda (J kg-1)'] = lmbda

    if write_spectral_files:
        scalar_df['wT'] = wT
        scalar_df['H'] = rho_cp * wT
        scalar_df['RH'] = rh_meteo * 100.

    # cospectra
    # linear frequency axis for cospectrum
    f = np.arange(0, np.ceil(nsoll / 2) + 1) / file_length
    if write_spectral_files:
        plot_cospectrum = cfg.get('plot', 'cospectrum')
        plot_spectrum = plot_base if plot_cospectrum else ''
        # spectrum for T
        spec_T, spec_T_scaled, _ = cospectrum(
            cfg, df['T_prime'], df['T_prime'], f, ws_mean,
            spectrum_type='spec', plot=plot_spectrum, tag='T')
        # cospectrum for wT
        cospec_wT, cospec_wT_scaled, _ = cospectrum(
            cfg, df['w_prime'], df['T_prime'], f, ws_mean,
            spectrum_type='cospec', plot=plot_spectrum, tag='wT')

        spectral_df['freq'] = log_bin_spectrum(f, np.zeros(len(f)),
                                               num_freq_bins, f[1], f[-1])[0]
        spectral_df['cospec_wT_scaled'] = cospec_wT_scaled

    # steady state test for wT according to Foken and Wichura 1996
    steady_state_wT_FW96 = (
        test_steady_state_FW96(df['w_prime'], df['T_prime']) * 100.)

    odf['steady_state_wT_FokenWichura1996 (%)'] = steady_state_wT_FW96

    # steady-state flagging (Mauder et al. AFM 2013)
    odf['H_qc'] = 0
    if (steady_state_wT_FW96 > 30.) and (steady_state_wT_FW96 < 100.):
        odf['H_qc'] = 1
    elif steady_state_wT_FW96 >= 100.:
        odf['H_qc'] = 2

    # Obukhov length, L
    L = -T_meteo * u_star**3 / (vonKarman * g * wT)

    # stability parameter, z/L
    sonic_height = cfg.get('sonic', 'sensor_height')
    zoL = sonic_height / L

    odf['L (m)'] = L
    odf['z/L'] = zoL

    if write_spectral_files:
        scalar_df['z/L'] = zoL

    # potential temperature, theta
    # virtual potential temperature, theta_v
    # and associated wT covariances
    p0 = 100000.  # reference pressure for potential temperature [Pa]
    pp = (p0 / p_meteo)**0.286
    # theta_mean = df['T_f'].mean() * pp
    df['theta_prime'] = df['T_prime'] * pp
    # wtheta = cov(df, 'w_prime', 'theta_prime')
    # note that H2O data is given in permille
    df['theta_v'] = df['T'] * pp * (1. + 0.38 * 1.e-3 * df['h2o_f'])
    if detrend_T:
        df['theta_v_prime'] = nandetrend(df['theta_v'])
    else:
        df['theta_v_prime'] = df['theta_v'] - df['theta_v'].mean()
    df['theta_v_prime'] = df['theta_v_prime'].fillna(0.)
    # wtheta_v = cov(df, 'w_prime', 'theta_v_prime')

    # test on developed turbulent conditions
    latitude = cfg.get('site', 'latitude')
    itc_w, itc_u, itc_T = test_itc(df['w_prime'], df['u_prime'], df['T_prime'],
                                   wT, zoL, u_star, latitude)
    itc_w *= 100.
    itc_u *= 100.
    itc_T *= 100.

    odf['itc_w (%)'] = itc_w
    odf['itc_u (%)'] = itc_u
    odf['itc_T (%)'] = itc_T

    if write_spectral_files:
        scalar_df['itc_w'] = itc_w / 100.

    # ITC flagging (Mauder et al. AFM 2013)
    itc_flag = 0
    if (itc_w > 30.) and (itc_w < 100.):
        itc_flag = 1
    elif itc_w >= 100.:
        itc_flag = 2

    if (itc_u > 30.) and (itc_u < 100.):
        if itc_flag < 2:
            itc_flag = 1
    elif itc_u >= 100.:
        itc_flag = 2

    if itc_flag == 1:
        if odf['H_qc'].iloc[0] < 2:
            odf['H_qc'] = 1
    elif itc_flag == 2:
        odf['H_qc'] = 2

    # residual vertical wind flagging (Mauder et al. AFM 2013)
    w_res = abs(df['w_rot'].mean())
    if (w_res > 0.1) and (w_res < 0.15):
        odf['H_qc'] = min(odf['H_qc'].iloc[0] + 1, 2)
    elif w_res >= 0.15:
        odf['H_qc'] = 2

    #
    # irga data
    #

    for TT in ['CO2', 'H2O']:
        Tname = f'{TT}_dry'
        tt = TT.lower()

        # interpolate
        df[f'{tt}_f'] = df[Tname].interpolate(method='nearest')

        # Reynolds decomposition
        if detrend_irga:
            df[f'{tt}_prime'] = nandetrend(df[f'{tt}_f'])
        else:
            df[f'{tt}_prime'] = df[f'{tt}_f'] - df[f'{tt}_f'].mean()
        if df[f'{tt}_prime'].isna().any():
            print(f'        NaN in {TT} detected',
                  df[f'{tt}_prime'].loc[df[f'{tt}_prime'].isna()])
            continue

        # covariances and lag times
        plot_covariance_peak = cfg.get('plot', 'covariance_peak')
        plot_covariance = (plot_base, tt) if plot_covariance_peak else tuple()
        wc, lag_samples = lagged_covariance(
            tt, cfg, w_prime=df['w_prime'], c_prime=df[f'{tt}_prime'],
            timelag_dict=timelag_dict, rh=rh_meteo,
            plot_label=plot_covariance)

        # align c_prime for time-lag and trim the circular part,
        # corrected time-series being needed for further computations
        c_prime = np.roll(df[f'{tt}_prime'].to_numpy(), lag_samples)
        w_prime = df['w_prime'].to_numpy()
        if lag_samples > 0:
            # Shift c_prime forward (to the right), remove front (wrapped)
            c_prime = c_prime[lag_samples:]
            # Remove first lag_samples from w_prime to align it in time
            w_prime = w_prime[lag_samples:]
        elif lag_samples < 0:
            lag_abs = abs(lag_samples)
            # Shift c_prime backward (to the left), remove end (wrapped)
            c_prime = c_prime[:-lag_abs]
            # Remove last lag_abs from w_prime to align
            w_prime = w_prime[:-lag_abs]

        # calculate spectrum for c and cospectrum for wc
        if write_spectral_files:
            f = np.arange(0, np.ceil(nsoll / 2) + 1) / file_length
            spec_c, spec_c_scaled, _ = cospectrum(
                cfg, c_prime, c_prime, f, ws_mean, spectrum_type='spec',
                plot=plot_spectrum, tag=f"c_{TT}")
            cospec_wc, cospec_wc_scaled, _ = cospectrum(
                cfg, w_prime, c_prime, f, ws_mean, spectrum_type='cospec',
                plot=plot_spectrum, tag=f"wc_{TT}")

            spectral_df[f'spec_{tt}_scaled'] = spec_c_scaled
            spectral_df[f'cospec_wc_{tt}_scaled'] = cospec_wc_scaled

        # steady state test for wc according to Foken and Wichura 1996
        steady_state_wc_FW96 = test_steady_state_FW96(w_prime, c_prime) * 100.

        odf[f'steady_state_wc_{tt}_FokenWichura1996 (%)'] = steady_state_wc_FW96

        if write_spectral_files:
            steady_state_wc_M98 = test_steady_state_M98(w_prime, c_prime)
            scalar_df[f'steady_state_wc_{tt}_Mahrt1998'] = steady_state_wc_M98

        # steady-state flagging (Mauder et al. AFM 2013)
        odf[f'F_{tt}_qc'] = 0
        if (steady_state_wc_FW96 > 30.) and (steady_state_wc_FW96 <= 100.):
            odf[f'F_{tt}_qc'] = 1
        elif steady_state_wT_FW96 > 100.:
            odf[f'F_{tt}_qc'] = 2

        # ITC flagging (Mauder et al. AFM 2013)
        if itc_flag == 1:
            if odf[f'F_{tt}_qc'].iloc[0] < 2:
                odf[f'F_{tt}_qc'] = 1
        elif itc_flag == 2:
            odf[f'F_{tt}_qc'] = 2

        # residual vertical wind flagging (Mauder et al. AFM 2013)
        if (w_res > 0.1) and (w_res < 0.15):
            odf[f'F_{tt}_qc'] = min(odf[f'F_{tt}_qc'].iloc[0] + 1, 2)
        elif w_res >= 0.15:
            odf[f'F_{tt}_qc'] = 2

        # compute flux uncertainties
        [flux_noise_mean, flux_noise_std, flux_noise_rmse,
         random_error_FS, random_flux, random_error_noise] = (
             flux_uncertainties(cfg, w_prime, c_prime))

        if tt == 'co2':
            # CO2 is in umol/mol
            mol = 'umol'
        elif tt == 'h2o':
            # H2O is in mmol/mol
            mol = 'mmol'
        else:
            raise ValueError(f'Gas not known: {TT}')
        odf[f'wc_{tt} ({mol} mol-1 m s-1)'] = wc
        odf[f'lag_samples_{tt} (#)'] = lag_samples
        odf[f'lag_time_{tt} (s)'] = lag_samples / sampling_rate
        odf[f'F_{tt} ({mol} m-2 s-1)'] = wc / Vd
        odf[f'F_{tt}_noise_mean ({mol} m-2 s-1)'] = flux_noise_mean / Vd
        odf[f'F_{tt}_noise_std ({mol} m-2 s-1)'] = flux_noise_std / Vd
        odf[f'F_{tt}_noise_rmse ({mol} m-2 s-1)'] = flux_noise_rmse / Vd
        odf[f'F_{tt}_random_error_FinkelsteinSims2001 ({mol} m-2 s-1)'] = (
            random_error_FS / Vd)
        odf[f'F_{tt}_random_flux ({mol} m-2 s-1)'] = random_flux / Vd
        odf[f'F_{tt}_random_error_noise ({mol} m-2 s-1)'] = (
            random_error_noise / Vd)

        if write_spectral_files:
            scalar_df[f'F_{tt}'] = wc / Vd

    # interdependencies of flags (Mauder et al. AFM 2013)
    if odf['F_h2o_qc'].iloc[0] == 2:
        odf['H_qc'] = min(odf['H_qc'].iloc[0] + 1, 2)
    if odf['H_qc'].iloc[0] == 2:
        odf['F_h2o_qc'] = min(odf['F_h2o_qc'].iloc[0] + 1, 2)
    if (odf['F_h2o_qc'].iloc[0] == 2) or (odf['H_qc'].iloc[0] == 2):
        odf['F_co2_qc'] = min(odf['F_co2_qc'].iloc[0] + 1, 2)

    # evapotranspiration and latent heat
    odf['E (kg m-2 s-1)'] = odf['F_h2o (mmol m-2 s-1)'] * 1.e-3 * Mv
    odf['E_qc'] = odf['F_h2o_qc']
    odf['LE (W m-2)'] = odf['E (kg m-2 s-1)'] * odf['lambda (J kg-1)']
    odf['LE_qc'] = odf['F_h2o_qc']

    if write_spectral_files:
        scalar_df['LE'] = odf['LE (W m-2)']

    # append to output file
    _save_df(odf, output_file)

    if write_spectral_files:
        sdf['scalar'] = scalar_df
        sdf['instrument'] = instrument_df
        sdf['spectral'] = spectral_df
        output_path = os.path.dirname(output_file)
        spectral_base = f'{output_path}/spectral/spectral'
        file_date_format = cfg.get('sonic', 'file_date_format')
        idate = get_dates_from_files(os.path.basename(ifile),
                                     file_date_format)[0]
        spectral_file = f'{spectral_base}_{idate.strftime("%Y%m%d%H%M%S")}.pkl'
        pickle.dump(sdf, open(spectral_file, 'wb'))

    return
