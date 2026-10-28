#!/usr/bin/env python
"""
Read different kind of files

This module was written by Matthias Cuntz while at Institut
National de Recherche pour l'Agriculture, l'Alimentation et
l'Environnement (INRAE), Nancy, France, using some code of
Bernard Heinesch, University of Liege, Gembloux Agro-Bio Tech.

:copyright: Copyright 2025- Matthias Cuntz, Bernard Heinesch, see AUTHORS.rst for details.
:license: MIT License, see LICENSE for details.

.. moduleauthor:: Matthias Cuntz

The following functions are provided:

.. autosummary::
   read_csv
   read_ghg
   read_meteo
   read_planar_fit_eddypro
   read_raw_datafile
   read_slt_eddysoft
   read_slt_edisol
   read_timelag
   read_timelag_eddypro
   read_timelag_file

History
   * Written Sep 2026 by Matthias Cuntz, using some code of
     read_metadata_files.py of GEddySoft (read_planar_fit)

"""
from collections.abc import Iterable
import os
import re
import zipfile
import numpy as np
import pandas as pd
from .config import read_config
from .get_files import get_files, get_dates_from_files


__all__ = ['read_csv', 'read_ghg', 'read_meteo', 'read_planar_fit_eddypro',
           'read_raw_datafile', 'read_slt_eddysoft', 'read_slt_edisol',
           'read_timelag', 'read_timelag_eddypro', 'read_timelag_file']


def _bittest(num, pos):
    '''
    Test if bit is 1 at position

    Parameters
    ----------
    num : int
        Number which bit pattern will be examined
    pos : int
        Number of bit to examine

    Returns
    -------
    bool
       True if bit at position pos is 1, else False

    '''
    # https://stackoverflow.com/questions/15876136/btest-function-in-fortran-is-there-any-equivalent-in-python
    return bool(num & (1 << pos))


def read_csv(ifile, csvargs={}):
    '''
    Read values from csv file(s) with pandas.read_csv

    Parameters
    ----------
    ifile : (iterable of) str
        csv file(s). Data will be concatenated if iterable.
    csvargs : dict, optional
        Dictionary setting or replacing pandas.read_csv keyword arguments
        Default: {'sep': ',',
                  'header': 'infer',
                  'skiprows': None,
                  'index_col': 0,
                  'parse_dates': True,
                  'na_values': '-9999',}

    Returns
    -------
    pandas.DataFrame

    '''
    if isinstance(ifile, Iterable) and (not isinstance(ifile, str)):
        ifiles = ifile
    else:
        ifiles = [ifile]

    ropt = {'sep': ',',
            'header': 'infer',
            'skiprows': None,
            'index_col': 0,
            'parse_dates': True,
            'na_values': '-9999',}
    for cc in csvargs:
        ropt[cc] = csvargs[cc]

    # set index_col manually to keep it for indexes
    if 'index_col' in ropt:
        index_col = ropt['index_col']
        del ropt['index_col']
    else:
        index_col = None
    if 'date_format' in ropt:
        date_format = ropt['date_format']
    else:
        date_format = 'ISO8601'

    dfl = []
    for ifil in ifiles:
        df = pd.read_csv(ifil, **ropt)
        if index_col is not None:
            if isinstance(index_col, str):
                iname = index_col
            else:
                iname = df.columns[index_col]
            df[iname] = pd.to_datetime(df[iname], format=date_format)
            df.set_index(iname, drop=False, inplace=True)
        dfl.append(df)

    # Concat datetime
    df = dfl[0]
    for df1 in dfl[1:]:
        df1 = df1[~df1.index.isin(df.index)]
        df, df1 = df.align(df1, axis=1)
        df = pd.concat([df, df1], verify_integrity=False)

    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]

    return df


def read_ghg(ifile, config):
    '''
    Read Licor ghg file

    Parameters
    ----------
    ifile : str
        Filename of ghg file
    config : str or astConfigParser
        Config (filename)

    Returns
    -------
    pandas.DataFrame

    '''
    cfg = read_config(config)

    sonic_columns = cfg.get('sonic', 'columns')
    sonic_names = [ s.strip() for s in cfg.get('sonic', 'names').split(',') ]
    irga_columns = cfg.get('irga', 'columns')
    irga_names = [ s.strip() for s in cfg.get('irga', 'names').split(',') ]

    with zipfile.ZipFile(ifile) as zfile:
        dfile = ''
        for nn in zfile.namelist():
            if nn.endswith('.data') and ('/' not in nn):
                if dfile == '':
                    dfile = nn
                else:
                    ValueError(f'Several .data files in {ifile}:\n'
                               f'{zfile.namelist()}')
        if dfile == '':
            ValueError(f'No .data file found in {ifile}:\n'
                       f'{zfile.namelist()}')

        with zfile.open(dfile) as ff:
            idf = pd.read_csv(ff, sep='\t', skiprows=7,
                              na_values='-9999')
            
    df_time = idf[['Date', 'Time']]
    # time format of ghg file is 12:30:01:005
    # posix/python format is 12:30:01.005
    df_time['Time'] = [ a[:a.rfind(':')] + '.' + a[a.rfind(':') + 1:]
                        for a in df_time['Time'] ]
    df_time['datetime'] = pd.to_datetime(df_time['Date'] + 'T' +
                                         df_time['Time'])
    df_time.drop(columns=['Date', 'Time'], inplace=True)
        
    dfl = [df_time,]

    if sonic_columns is not None:
        df_sonic = idf.iloc[:, list(sonic_columns)]

        if sonic_names is not None:
            cols = list(df_sonic.columns)
            sdict = {}
            for i in range(len(sonic_columns)):
                sdict.update({cols[i]: sonic_names[i]})
            df_sonic.rename(columns=sdict, inplace=True)

        dfl.append(df_sonic)

    if irga_columns is not None:
        df_irga = idf.iloc[:, list(irga_columns)]
        
        if irga_names is not None:
            cols = list(df_irga.columns)
            idict = {}
            for i in range(len(irga_columns)):
                idict.update({cols[i]: irga_names[i]})
            df_irga.rename(columns=idict, inplace=True)

        dfl.append(df_irga)

    df = pd.DataFrame()
    for odf in dfl:
        df = pd.concat([df, odf], axis=1)

    # df['timestamp'] = [ dd.timestamp() for dd in df['datetime'] ]
    df.set_index('datetime', inplace=True)

    return df


def read_meteo(config):
    '''
    Read meteo files given in config file

    Parameters
    ----------
    config : str or astConfigParser
        Config (filename)

    Returns
    -------
    pandas.DataFrame

    '''
    cfg = read_config(config)

    section = 'meteo'

    # read file(s)
    files = get_files(cfg, section)

    idate = cfg.get(section, 'date')
    dcsv = {}
    for oo in ['sep', 'date', 'date_format', 'na_values']:
        ocfg = cfg.get(section, oo)
        if oo:
            if oo == 'date':
                dcsv.update({'index_col': ocfg})
            else:
                dcsv.update({oo: ocfg})

    df = read_csv(files, csvargs=dcsv)

    # drop datetime column
    df.drop(columns=[idate,], inplace=True)

    return df


def read_planar_fit_eddypro(config):
    '''
    Read planar fit config file

    Parameters
    ----------
    config : str or astConfigParser
        Config (filename)

    Returns
    -------
    pandas.DataFrame

    '''
    cfg = read_config(config)

    section = 'rotation'

    # read file(s)
    files = get_files(cfg, section)

    # this is the file coming from the PFM preparation of EddyPro
    if len(files) > 1:
        print('Warning: multiple planar fit files not implemented yet.')
    ifile = files[0]
    with open(ifile, 'r') as ff:
        lines = ff.readlines()

    # Find the index of the line containing "Rotation matrices"
    rotation_matrices_index = lines.index('Rotation matrices\n') + 1

    R_tilt_PFM = {}
    sect = 1
    for i in range(rotation_matrices_index, len(lines), 4):
        # Split the text using one or more spaces as the delimiter
        elements = re.split(r'\s+', lines[i])
        # Filter out any empty strings resulting from consecutive spaces
        elements = [ element for element in elements if element ]

        sector = int(elements[4])

        matrix = []
        for j in range(i + 1, i + 4):
            row = list(map(float, lines[j].strip().split()))
            matrix.append(row)
        R_tilt_PFM[sector] = np.array(matrix)

        sect = sect + 1

    return R_tilt_PFM


def read_raw_datafile(ifile, config):
    '''
    Read raw data files such as Licor ghg or slt files

    Parameters
    ----------
    ifile : str
        Filename of ghg file
    config : str or astConfigParser
        Config (filename)

    Returns
    -------
    pandas.DataFrame

    '''
    cfg = read_config(config)

    file_type = cfg.get('sonic', 'file_type')
    if file_type.lower() == 'ghg':
        return read_ghg(ifile, config)
    elif file_type.lower() == 'slt_eddysoft':
        return read_slt_eddysoft(ifile, config)
    elif file_type.lower() == 'slt_edisol':
        return read_slt_edisol(ifile, config)
    else:
        raise ValueError(f'File type not implemented: {file_type}')

    return


def read_slt_eddysoft(ifile, config):
    '''
    Read slt file of Olaf Kolle's EddySoft/EddyMeas suite

    Parameters
    ----------
    ifile : str
        Filename of slt file
    config : str or astConfigParser
        Config (filename)

    Returns
    -------
    pandas.DataFrame

    '''
    cfg = read_config(config)

    sonic_columns = cfg.get('sonic', 'columns')
    sonic_names = [ s.strip() for s in cfg.get('sonic', 'names').split(',') ]
    irga_columns = cfg.get('irga', 'columns')
    irga_names = [ s.strip() for s in cfg.get('irga', 'names').split(',') ]

    with open(ifile, 'rb') as ff:
        # determine number of columns from first two bytes
        ibytes, iversion = np.fromfile(ff, dtype='<i1', count=2)
        ncol = ibytes // 2
        nanalog = ncol - 4  # 4 sonic signals
        assert nanalog == len(irga_columns), (
            f'Number or IRGA columns in input file ({nanalog})'
            f' do not match number of given IRGA columns: {irga_columns},'
            f' {irga_names}')

        # 3 dummy 16-bit values (O. Kolle's EddySoft code)
        _ = np.fromfile(ff, dtype='<i2', count=3)

        # information on analog signal, i.e. columns > 4
        ainfo = np.fromfile(ff, dtype='<i1', count=nanalog * 2)
        ares = ainfo[::2]
        # # EddyPro
        # # if mask byte is even, low resolution is used
        # # if mask byte is odd, high resolution is used
        # high_res = [ not (ares[i] == 0 or ares[i] // 2 * 2 == ares[i])
        #              for i in range(nanalog) ]
        # use high resolution if first bit set (from EddySoft code)
        high = [ _bittest(ares[i], 0) for i in range(nanalog) ]

        # read data
        edata = np.fromfile(ff, dtype='<i2')

    edata = edata.reshape(-1, ncol).astype(float)

    # make datetime index
    # idate = pd.to_datetime(os.path.basename(ifile)[1:12], format='%Y%j%H%M')
    file_date_format = cfg.get('sonic', 'file_date_format')
    idate = get_dates_from_files(os.path.basename(ifile), file_date_format)[0]
    nrow = edata.shape[0]
    sampling_rate = cfg.get('sonic', 'sampling_rate')
    msec = int(round(1. / sampling_rate * 1000.))  # ms
    idates = idate + np.timedelta64(msec, 'ms') * np.arange(nrow)

    df = pd.DataFrame({}, index=idates)
    df.index.name = 'datetime'

    windvars = ['u', 'v', 'w']
    if sonic_names[0] in windvars:
        df[sonic_names[0]] = edata[:, sonic_columns[0]] * 0.01  # m/s
    else:
        raise ValueError(f'1st sonic variable must be u, v, or w. Is: {sonic_names[0]}')
    if sonic_names[1] in windvars:
        df[sonic_names[1]] = edata[:, sonic_columns[1]] * 0.01  # m/s
    else:
        raise ValueError(f'2nd sonic variable must be u, v, or w. Is: {sonic_names[1]}')
    if sonic_names[2] in windvars:
        df[sonic_names[2]] = edata[:, sonic_columns[2]] * 0.01  # m/s
    else:
        raise ValueError(f'3rd sonic variable must be u, v, or w. Is: {sonic_names[2]}')

    if sonic_names[3].lower() == 'ts':
        tmean = (edata[:, sonic_columns[3]] / 100.).mean()
        if (tmean < 100.):    # degC
            df[sonic_names[3]] = edata[:, sonic_columns[3]] * 0.01
        elif (tmean > 200.):  # K
            df[sonic_names[3]] = edata[:, sonic_columns[3]] * 0.01 - 273.15
        else:                 # speed of sound, m/s
            df['sos'] = edata[:, sonic_columns[3]] * 0.02
            df[sonic_names[3]] = df['sos']**2 / 403. - 273.15
    elif sonic_names[3].lower() == 'sos':
        df[sonic_names[3]] = edata[:, sonic_columns[3]] * 0.02
        df['ts'] = df[sonic_names[3]]**2 / 403. - 273.15

    co2_gain = cfg.get('irga', 'co2_gain')
    co2_offset = cfg.get('irga', 'co2_offset')
    h2o_gain = cfg.get('irga', 'h2o_gain')
    h2o_offset = cfg.get('irga', 'h2o_offset')
    for i in range(nanalog):
        if high[i]:
            df[irga_names[i]] = (edata[:, irga_columns[i]] + 25000.) * 0.1
        else:
            df[irga_names[i]] = edata[:, irga_columns[i]]

        # # in EddySoft code
        # df[irga_names[i]] /= 5000.

        if irga_names[i].lower().startswith('co2'):
            df[irga_names[i]] = co2_gain * df[irga_names[i]] + co2_offset
        elif irga_names[i].lower().startswith('h2o'):
            df[irga_names[i]] = h2o_gain * df[irga_names[i]] + h2o_offset
        else:
            raise ValueError(f'Species not implemented: {irga_names[i]}')

    return df


def read_slt_edisol(ifile, config):
    '''
    Read slt file of EdiSol software

    Parameters
    ----------
    ifile : str
        Filename of slt file
    config : str or astConfigParser
        Config (filename)

    Returns
    -------
    pandas.DataFrame

    '''
    cfg = read_config(config)

    sonic_columns = cfg.get('sonic', 'columns')
    sonic_names = [ s.strip() for s in cfg.get('sonic', 'names').split(',') ]
    irga_columns = cfg.get('irga', 'columns')
    irga_names = [ s.strip() for s in cfg.get('irga', 'names').split(',') ]

    with open(ifile, 'rb') as ff:
        # determine number of columns from first byte
        rec_len = np.fromfile(ff, dtype='<i1', count=1)
        ncol = rec_len // 2
        nanalog = ncol - 4  # 4 sonic signals
        assert nanalog == len(irga_columns), (
            f'Number or IRGA columns in input file ({nanalog})'
            f' do not match number of given IRGA columns: {irga_columns},'
            f' {irga_names}')

        # read rest of header - ignore
        rec_len = np.fromfile(ff, dtype='<i1', count=rec_len - 1)

        # read data
        edata = np.fromfile(ff, dtype='<i2')

    edata = edata.reshape(-1, ncol).astype(float)

    # make datetime index
    # idate = pd.to_datetime(os.path.basename(ifile)[1:12], format='%Y%j%H%M')
    file_date_format = cfg.get('sonic', 'file_date_format')
    idate = get_dates_from_files(os.path.basename(ifile), file_date_format)[0]
    nrow = edata.shape[0]
    sampling_rate = cfg.get('sonic', 'sampling_rate')
    msec = round(1. / sampling_rate * 1000.).astype(int)  # ms
    idates = idate + np.timedelta64(msec, 'ms') * np.arange(nrow)

    df = pd.DataFrame({}, index=idates)
    df.index.name = 'datetime'

    windvars = ['u', 'v', 'w']
    if sonic_names[0] in windvars:
        df[sonic_names[0]] = edata[:, sonic_columns[0]] * 0.01  # m/s
    else:
        raise ValueError(f'1st sonic variable must be u, v, or w. Is: {sonic_names[0]}')
    if sonic_names[1] in windvars:
        df[sonic_names[1]] = edata[:, sonic_columns[1]] * 0.01  # m/s
    else:
        raise ValueError(f'2nd sonic variable must be u, v, or w. Is: {sonic_names[1]}')
    if sonic_names[2] in windvars:
        df[sonic_names[2]] = edata[:, sonic_columns[2]] * 0.01  # m/s
    else:
        raise ValueError(f'3rd sonic variable must be u, v, or w. Is: {sonic_names[2]}')

    if sonic_names[3].lower() == 'ts':
        tmean = (edata[:, sonic_columns[3]] / 100.).mean()
        if (tmean < 100.):    # degC
            df[sonic_names[3]] = edata[:, sonic_columns[3]] * 0.01
        elif (tmean > 200.):  # K
            df[sonic_names[3]] = edata[:, sonic_columns[3]] * 0.01 - 273.15
        else:                 # speed of sound, m/s
            df['sos'] = edata[:, sonic_columns[3]] * 0.02
            df[sonic_names[3]] = df['sos']**2 / 403. - 273.15
    elif sonic_names[3].lower() == 'sos':
        df[sonic_names[3]] = edata[:, sonic_columns[3]] * 0.02
        df['ts'] = df[sonic_names[3]]**2 / 403. - 273.15

    co2_gain = cfg.get('irga', 'co2_gain')
    co2_offset = cfg.get('irga', 'co2_offset')
    h2o_gain = cfg.get('irga', 'h2o_gain')
    h2o_offset = cfg.get('irga', 'h2o_offset')
    for i in range(nanalog):
        if irga_names[i].lower().startswith('co2'):
            df[irga_names[i]] = co2_gain * edata[:, irga_columns[i]] + co2_offset
        elif irga_names[i].lower().startswith('h2o'):
            df[irga_names[i]] = h2o_gain * edata[:, irga_columns[i]] + h2o_offset
        else:
            raise ValueError(f'Species not implemented: {irga_names[i]}')

    return df


def read_timelag(config):
    '''
    Read timelag file or EddyPro timelag optimisation file

    Parameters
    ----------
    config : str or astConfigParser
        Config (filename)

    Returns
    -------
    pandas.DataFrame

    '''
    cfg = read_config(config)

    section = 'timelag'

    imethod = cfg.get(section, 'method')

    if imethod.lower() == 'eddypro':
        return read_timelag_eddypro(config)
    elif imethod.lower() == 'file':
        return read_timelag_file(config)
    else:
        co2_center = cfg.get(section, 'co2_center')
        co2_size = cfg.get(section, 'co2_size')
        co2_filter_length = cfg.get(section, 'co2_filter_length')
        h2o_center = cfg.get(section, 'h2o_center')
        h2o_size = cfg.get(section, 'h2o_size')
        h2o_filter_length = cfg.get(section, 'h2o_filter_length')
        return (co2_center, co2_size, co2_filter_length,
                h2o_center, h2o_size, h2o_filter_length)


def read_timelag_eddypro(config):
    '''
    Read EddyPro timelag file

    Parameters
    ----------
    config : str or astConfigParser
        Config (filename)

    Returns
    -------
    pandas.DataFrame

    Notes
    -----
    EddyPro's covariance function moves the timeseries in the other
    direction than HEddySoft, i.e. positive timelags in EddyPro are negative
    timelags in HEddySoft and vice versa.

    '''
    cfg = read_config(config)

    section = 'timelag'

    # read file(s)
    files = get_files(cfg, section)

    # this is the file coming from timelag optimization of EddyPro
    if len(files) > 1:
        print('Warning: multiple EddyPro timelag files not implemented yet.')
    ifile = files[0]
    with open(ifile, 'r') as ff:
        lines = ff.readlines()
    nline = len(lines)

    # CO2
    undef = -9999
    co2med = undef
    co2min = undef
    co2max = undef

    for line in lines:
        if line.startswith('Median_co2_timelag'):
            co2med = float(re.split(r'\s+', line)[1])
        if line.startswith('Mimimum_co2_timelag'):
            co2min = float(re.split(r'\s+', line)[1])
        if line.startswith('Maximum_co2_timelag'):
            co2max = float(re.split(r'\s+', line)[1])

    # Swap positive to negative and vice versa
    co2med *= -1.
    co2min, co2max = -co2max, -co2min

    co2 = (co2med, co2min, co2max)
            
    # H2O
    # Find the index of the line containing "class"
    cline = -1
    for line in lines:
        cline += 1
        if line.startswith('class'):
            break
    nclass = nline - cline - 1

    iclass = np.full(nclass, int(undef))
    irh = np.full(nclass, float(undef))
    imed = np.full(nclass, float(undef))
    imin = np.full(nclass, float(undef))
    imax = np.full(nclass, float(undef))

    for i in range(nclass):
        iline = re.split(r'\s+', lines[cline + 1 + i])
        iclass[i] = int(iline[1])
        irh[i] = 0.5 * (float(iline[2]) + float(iline[4].replace('%', '')))
        imed[i] = float(iline[5])
        imin[i] = float(iline[6])
        imax[i] = float(iline[7])

    # Swap positive to negative and vice versa
    imed *= -1.
    imin, imax = -imax, -imin

    df = pd.DataFrame({
        "class": iclass,
        "rh": irh,
        "h2omed": imed,
        "h2omin": imin,
        "h2omax": imax,})
    df.set_index('class', inplace=True)
        
    return co2, df


def read_timelag_file(config):
    '''
    Read prescribed timelags

    Parameters
    ----------
    config : str or astConfigParser
        Config (filename)

    Returns
    -------
    pandas.DataFrame

    '''
    cfg = read_config(config)

    section = 'timelag'

    # read file(s)
    files = get_files(cfg, section)

    idate = cfg.get(section, 'date')
    dcsv = {}
    for oo in ['sep', 'date', 'date_format', 'na_values']:
        ocfg = cfg.get(section, oo)
        if oo:
            if oo == 'date':
                dcsv.update({'index_col': ocfg})
            else:
                dcsv.update({oo: ocfg})

    df = read_csv(files, csvargs=dcsv)

    # drop datetime column
    df.drop(columns=[idate,], inplace=True)

    # timelag
    ilagco2 = cfg.get(section, 'co2_timelag')
    ilagh2o = cfg.get(section, 'h2o_timelag')
    issecond = cfg.get(section, 'issecond')
    sampling_rate = cfg.get('sonic', 'sampling_rate')
    if issecond:
        df.rename(columns={ilagco2: 'co2_timelag_seconds',
                           ilagh2o: 'h2o_timelag_seconds'}, inplace=True)
        # make sure no NaN, which is not supported by astype(int)
        df['co2_timelag_seconds'] = df['co2_timelag_seconds'].interpolate(
            method='linear')
        df['h2o_timelag_seconds'] = df['h2o_timelag_seconds'].interpolate(
            method='linear')
        df['co2_timelag_index'] = (df['co2_timelag_seconds'] *
                                   sampling_rate).round().astype(int)
        df['h2o_timelag_index'] = (df['h2o_timelag_seconds'] *
                                   sampling_rate).round().astype(int)
    else:
        df.rename(columns={ilagco2: 'co2_timelag_index',
                           ilagh2o: 'h2o_timelag_index'}, inplace=True)
        df['co2_timelag_index'] = df['co2_timelag_index'].interpolate(
            method='nearest')
        df['h2o_timelag_index'] = df['h2o_timelag_index'].interpolate(
            method='nearest')
        df['co2_timelag_seconds'] = df['co2_timelag_index'] / sampling_rate
        df['h2o_timelag_seconds'] = df['h2o_timelag_index'] / sampling_rate

    return df
