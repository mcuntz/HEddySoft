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
# import sys
import multiprocessing as mp
import pandas as pd
from .argsort import argsort
from .config import read_config, get_config_base_dir
from .get_files import get_files, get_dates_from_files
from .read_files import read_meteo, read_planar_fit_eddypro, read_timelag
from .heddysoft_main import heddysoft_main


__all__ = ['heddysoft']


def heddysoft(cfgfile):
    # get ini information
    cfg = read_config(cfgfile)

    # checks
    # only ghg implemented yet
    ftype = cfg.get('sonic', 'file_type')
    if ftype.lower() != 'ghg':
        raise ValueError(f'Only ghg file type implemented.'
                         f' Given: {ftype}')
    
    # only ghg implemented yet so sonic and irga data in same file
    idir = cfg.get('irga', 'dir')
    if idir:
        raise ValueError(f'Currently, sonic and irga files have to be'
                         f' the same. Do not give "dir" in section "irga".'
                         f' Currently given: {idir}')
    
    # get sonic files (irga files have to match later)
    allfiles = get_files(cfg, 'sonic')
    alldates = get_dates_from_files(allfiles)
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
        print(f'Files to process: {files}')

    # output path
    base_dir = get_config_base_dir(cfg)
    output_path = cfg.get('files', 'output_dir')
    if not output_path.startswith('/'):
        output_path = f'{base_dir}/{output_path}'
    os.makedirs(output_path, exist_ok=True)
    output_file = cfg.get('files', 'output_file')
    output_file = f'{output_path}/{output_file}'

    # output path if any plotting True
    idoplot = False
    for key in cfg['plot']:
        doplot = cfg.get('plot', key)
        if isinstance(doplot, bool):
            if doplot:
                idoplot = True
    if idoplot:
        ppath = cfg.get('plot', 'dir')
        if not ppath.startswith('/'):
            ppath = f'{base_dir}/{ppath}'
        os.makedirs(ppath, exist_ok=True)

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

    # Process the files
    start_time = datetime.datetime.now()
    print(f'\n  Start processing at'
          f' {start_time.strftime("%Y-%m-%d %H:%M:%S")}')

    # pass also config to heddysoft
    party  = partial(heddysoft_main, cfg, df_meteo, output_file,
                     timelag_dict=timelag_dict,
                     rotation_matrix=rotation_matrix)

    if cfg.get('param', 'multiprocessing'):

        # original_stdout = sys.stdout
        with mp.Pool(mp.cpu_count()) as pool:
            try:
                # # redirect stdout to /dev/null for all processes
                # sys.stdout = open(os.devnull, 'w')                
                pool.map(party, files)

                # # restore stdout
                # sys.stdout = sys.__stdout__

                print(f'\n  Multiprocessed files: {files}')

            except Exception as e:
                # # make sure to restore stdout even if there's an error
                # sys.stdout = original_stdout
                print(f'  Error during processing: {str(e)}')

            finally:
                pool.close()
                pool.join()
                print('  Pool closed and joined')

    else:
        for ff in files:
            party(ff)
        print(f'\n  Processed files: {files}')
    calc_end_time = datetime.datetime.now()
    print(f'    time: {calc_end_time - start_time}')
    
    print(f'\n  Read and sort {outputfile}')
    sort_start_time = datetime.datetime.now()
    df = pd.read_csv(output_file, sep=',', index_col=0,
                     parse_dates=True, date_format='ISO8601')
    df.sort_index(inplace=True)
    df.index.name = 'TIMESTAMP_START'
    print(f'  Rewrite {output_file}')
    df.to_csv(output_file, na_rep=-9999)
    sort_end_time = datetime.datetime.now()
    print(f'    time: {sort_end_time - sort_start_time}')

    end_time = datetime.datetime.now()
    print(f'\nFinished at {end_time.strftime("%Y-%m-%d %H:%M:%S")}'
          f' taking time {end_time - start_time}')
