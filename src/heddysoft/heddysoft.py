#!/usr/bin/env python
"""
HEddySoft startup

This module was written by Matthias Cuntz while at Department of
Computational Hydrosystems, Helmholtz Centre for Environmental
Research - UFZ, Leipzig, Germany, and continued while at Institut
National de Recherche pour l'Agriculture, l'Alimentation et
l'Environnement (INRAE), Nancy, France.

:copyright: Copyright 2026 Matthias Cuntz, see AUTHORS.rst for details.
:license: MIT License, see LICENSE for details.

.. moduleauthor:: Matthias Cuntz

The following functions are provided

.. autosummary::
   heddysoft

History
   * Written Sep 2026 by Matthias Cuntz (mc (at) macu (dot) de)
     from GEddySoft_start.py of GEddySoft of Bernard Heinesch

"""
import datetime
from functools import partial
import os
import pickle
import multiprocessing as mp
import numpy as np
from joblib import Parallel, delayed
import pandas as pd
from .argsort import argsort
from .config import read_config, get_config_base_dir
from .get_files import get_files, get_dates_from_files
from .read_files import read_meteo, read_planar_fit_eddypro, read_timelag
from .heddysoft_main import heddysoft_main
from .heddysoft_timelag import heddysoft_timelag


__all__ = ['heddysoft']


def _append(odf, df, df_col, ocol, idf):
    """
    Append spectral DataFrame to output DataFrame

    Parameters
    ----------
    odf : pandas.DataFrame
        DataFrame to append to
    df : pandas.DataFrame
        DataFrame from which column df_col will be extracted
    df_col : str
        Column of df
    ocol : list of str
        Output column names
    idf : pandas.DataFrame
        DataFrame for index

    Returns
    -------
    None

    """
    iodf = pd.DataFrame([df[df_col].to_numpy()],
                        columns=ocol, index=idf.index)
    iodf.index.name = idf.index.name

    return pd.concat([odf, iodf])


def heddysoft(config, dotimelag=False):
    """
    Start HEddySoft

    Parameters
    ----------
    config : str or astConfigParser
        Config (filename)
    dotimelag : bool, optional
        If True, only calculate timelags

    Returns
    -------
    None

    """
    # get ini information
    cfg = read_config(config)

    #
    # Checks
    #

    # only ghg implemented yet
    ftype = cfg.get('sonic', 'file_type')
    if not ((ftype.lower() == 'ghg') or ftype.lower().startswith('slt')):
        raise ValueError(f'Only ghg and slt file types implemented.'
                         f' Given: {ftype}')

    # only ghg implemented yet, i.e. sonic and irga data in same file
    idir = cfg.get('irga', 'dir')
    if idir:
        raise ValueError(f'Currently, sonic and irga files have to be'
                         f' the same. Do not give "dir" in section "irga".'
                         f' Currently given: {idir}')

    #
    # Get files and prepare output
    #

    # get sonic files
    allfiles = get_files(cfg, 'sonic')
    file_date_format = cfg.get('sonic', 'file_date_format')
    alldates = get_dates_from_files(allfiles, file_date_format)
    ii = argsort(alldates)
    allfiles = [ allfiles[i] for i in ii ]
    alldates = [ alldates[i] for i in ii ]

    # select files in time
    start_date = cfg.get('dates', 'start_date')
    end_date = cfg.get('dates', 'end_date')
    if start_date:
        sdate = pd.to_datetime(start_date, format='ISO8601')
    else:
        sdate = alldates[0]
    if end_date:
        edate = pd.to_datetime(end_date, format='ISO8601')
    else:
        edate = alldates[-1]
    files = [ allfiles[ii] for ii, dd in enumerate(alldates)
              if (dd >= sdate) and (dd <= edate) ]

    if len(files) == 0:
        print(f'\nNo files to process between {start_date} and {end_date}')
        return
    else:
        idir = os.path.dirname(files[0])
        filenames = [ os.path.basename(ff) for ff in files ]
        print(f'{len(filenames)} files to process in directory {idir}:'
              f' {filenames[0]} to {filenames[-1]}')

    # output path
    base_dir = get_config_base_dir(cfg)
    output_path = cfg.get('files', 'output_dir')
    if not (output_path.startswith('/') or (output_path[1] == ':')):
        output_path = f'{base_dir}/{output_path}'
    os.makedirs(output_path, exist_ok=True)

    output_file = cfg.get('files', 'output_file')
    output_base = output_file[:output_file.rfind('.')]
    output_end = output_file[output_file.rfind('.'):]
    output_file = (f'{output_path}/{output_base}'
                   f'_{sdate.strftime("%Y%m%d%H%M%S")}'
                   f'-{edate.strftime("%Y%m%d%H%M%S")}{output_end}')

    # output path if any plotting True
    idoplot = False
    for key in cfg['plot']:
        doplot = cfg.get('plot', key)
        if isinstance(doplot, bool):
            if doplot:
                idoplot = True
    if idoplot:
        ppath = cfg.get('plot', 'dir')
        if not (ppath.startswith('/') or (ppath[1] == ':')):
            ppath = f'{base_dir}/{ppath}'
        os.makedirs(ppath, exist_ok=True)

    #
    # Read ancillary files
    #

    # read meteo file
    df_meteo = read_meteo(cfg)

    # read meteo file
    lag_method = cfg.get('timelag', 'method')
    timelag_dict = {'method': lag_method.lower()}
    res = read_timelag(cfg)
    if lag_method.lower() == 'eddypro':
        timelag_dict.update({'co2_timelag': res[0],
                             'h2o_timelag': res[1]})
    elif lag_method.lower() == 'file':
        timelag_dict.update({'timelags': res})
    else:
        timelag_dict.update({'co2_center': res[0],
                             'co2_size': res[1],
                             'co2_filter_length': res[2],
                             'h2o_center': res[3],
                             'h2o_size': res[4],
                             'h2o_filter_length': res[5]})

    # read planar fit file
    rotation_mode = cfg.get('rotation', 'mode')
    if rotation_mode == 2:
        rotation_matrix = read_planar_fit_eddypro(cfg)
    else:
        rotation_matrix = None

    #
    # Process files
    #

    start_time = datetime.datetime.now()
    print(f'\n  Start processing at'
          f' {start_time.strftime("%Y-%m-%d %H:%M:%S")}')
    if dotimelag:
        print('  Do timelags only')
        iprog = heddysoft_timelag
    else:
        iprog = heddysoft_main
    party = partial(iprog, cfg, df_meteo, output_file,
                    timelag_dict=timelag_dict,
                    rotation_matrix=rotation_matrix)
    if cfg.get('param', 'multiprocessing'):
        Parallel(n_jobs=mp.cpu_count()-2)(delayed(party)(ff)
                                          for ff in files)
        print(f'\n  Multiprocessed {len(filenames)} files on'
              f' {mp.cpu_count()-2} cpus.')
    else:
        for ff in files:
            party(ff)
        print(f'\n  Processed {len(filenames)} files.')
    calc_end_time = datetime.datetime.now()
    print(f'      time: {calc_end_time - start_time}')

    #
    # Postprocess
    #

    # Sort output file
    print('\n  Read, sort, rewrite')
    print(f'      {output_file}')
    sort_start_time = datetime.datetime.now()
    df = pd.read_csv(output_file, sep=',', index_col=0,
                     parse_dates=True, date_format='ISO8601',
                     on_bad_lines='warn')
    df.sort_index(inplace=True)
    df.index.name = 'TIMESTAMP_START'

    # drop duplicates and fill missing times
    file_length = cfg.get('sonic', 'file_length')
    # -> do not use df.drop_duplicates(inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    df = df.resample(f'{file_length}s').asfreq(np.nan)

    df.to_csv(output_file, na_rep=-9999)

    # Sort spectral files
    write_spectral_files = cfg.get('param', 'write_spectral_files')
    if write_spectral_files:
        spectral_file = (f'{output_path}/{output_base}'
                         f'_{sdate.strftime("%Y%m%d%H%M%S")}'
                         f'-{edate.strftime("%Y%m%d%H%M%S")}-spectral.pkl')
        print(f'      {spectral_file}')
        sfiles = get_files(cfg, 'spectral')
        sdates = get_dates_from_files(sfiles, '%Y%m%d%H%M%S')
        sfiles = [ sfiles[ii] for ii, dd in enumerate(sdates)
                   if (dd >= sdate) and (dd <= edate) ]

        sdf = pd.DataFrame()
        ipt_w = pd.DataFrame()
        ipt_co2 = pd.DataFrame()
        ipt_h2o = pd.DataFrame()
        freq = pd.DataFrame()
        icon = pd.DataFrame()
        rcon_co2 = pd.DataFrame()
        rcon_h2o = pd.DataFrame()
        rcon_wc_co2 = pd.DataFrame()
        rcon_wc_h2o = pd.DataFrame()
        for ff in sfiles:
            ldf = pickle.load(open(ff, 'rb'))

            # ['TIMESTAMP_END', 'wT', 'wind_speed', 'wind_direction', ...]
            scalar_df = ldf['scalar']
            sdf = pd.concat([sdf, scalar_df])

            # ipt_w, ipt_co2, ipt_h2o - 12 columns
            instrument_df = ldf['instrument']
            icol = range(12)

            ipt_w = _append(ipt_w, instrument_df, 'ipt_w', icol, scalar_df)
            ipt_co2 = _append(ipt_co2, instrument_df, 'ipt_co2', icol, scalar_df)
            ipt_h2o = _append(ipt_h2o, instrument_df, 'ipt_h2o', icol, scalar_df)

            # freq, cospec_wT_scaled, spec_co2_scaled, cospec_wc_co2_scaled,
            # spec_h2o_scaled, cospec_wc_h2o_scaled - 50 columns
            spectral_df = ldf['spectral']
            icol = spectral_df['freq'].values

            freq = _append(freq, spectral_df, 'freq', icol, scalar_df)
            icon = _append(icon, spectral_df, 'cospec_wT_scaled', icol, scalar_df)
            rcon_co2 = _append(rcon_co2, spectral_df, 'spec_co2_scaled',
                               icol, scalar_df)
            rcon_h2o = _append(rcon_h2o, spectral_df, 'spec_h2o_scaled',
                               icol, scalar_df)
            rcon_wc_co2 = _append(rcon_wc_co2, spectral_df, 'cospec_wc_co2_scaled',
                                  icol, scalar_df)
            rcon_wc_h2o = _append(rcon_wc_h2o, spectral_df, 'cospec_wc_h2o_scaled',
                                  icol, scalar_df)

        sdf.sort_index(inplace=True)
        ipt_w.sort_index(inplace=True)
        ipt_co2.sort_index(inplace=True)
        ipt_h2o.sort_index(inplace=True)
        freq.sort_index(inplace=True)
        icon.sort_index(inplace=True)
        rcon_co2.sort_index(inplace=True)
        rcon_h2o.sort_index(inplace=True)
        rcon_wc_co2.sort_index(inplace=True)
        rcon_wc_h2o.sort_index(inplace=True)

        sout = {'scalar': sdf,
                'ipt_w': ipt_w, 'ipt_co2': ipt_co2, 'ipt_h2o': ipt_h2o,
                'freq': freq,
                'cospec_wT': icon, 'spec_co2': rcon_co2, 'spec_h2o': rcon_h2o,
                'cospec_wc_co2': rcon_wc_co2, 'cospec_wc_h2o': rcon_wc_h2o}
        pickle.dump(sout, open(spectral_file, 'wb'))

        for ff in sfiles:
            os.remove(ff)

    sort_end_time = datetime.datetime.now()
    print(f'      time: {sort_end_time - sort_start_time}')

    #
    # Finished
    #

    end_time = datetime.datetime.now()
    print(f'\nFinished at {end_time.strftime("%Y-%m-%d %H:%M:%S")}'
          f' taking time {end_time - start_time}')
