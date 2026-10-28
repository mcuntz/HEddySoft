#!/usr/bin/env python
"""
Dealing with files and filenames

This module was written by Matthias Cuntz while at
Institut National de Recherche pour l'Agriculture, l'Alimentation et
l'Environnement (INRAE), Nancy, France.

:copyright: Copyright 2026- Matthias Cuntz, see AUTHORS.rst for details.
:license: MIT License, see LICENSE for details.

.. moduleauthor:: Matthias Cuntz

The following functions are provided:

.. autosummary::
   get_files
   get_dates_from_files

History
   * Written, Sep 2026, Matthias Cuntz
   * Bug: directories starting with / are not prepended with basedir,
     Oct 2026, Matthias Cunt

"""
import glob
import os
import re
import pandas as pd
from .config import get_config_base_dir, read_config


__all__ = ['get_files', 'get_dates_from_files']


def get_files(config, ftype):
    """
    Read files from directories and filenames given in config file
    
    Parameters
    ----------
    config : str or astConfigParser
        Config (filename)
    ftype : str
        File type 'sonic', 'irga', 'meteo', 'timelag', 'rotation'

    Returns
    -------
    list
        List of filenames with absolute pathes

    """
    cfg = read_config(config)
    base_dir = get_config_base_dir(cfg)

    ltype = ftype.lower()
    if ltype not in ['sonic', 'irga', 'meteo', 'timelag', 'rotation']:
        raise ValueError(f'ftype not known: {ftype}')

    section = ltype
    idirs = cfg.get(section, 'dir')
    ifile = cfg.get(section, 'files')
    
    if isinstance(idirs, str):
        idirs = [idirs]

    odirs = []
    for idir in idirs:
        if (idir.startswith('/') or (idir[1] == ':')):
            odirs.append(f'{idir}')
        else:
            odirs.append(f'{base_dir}/{idir}')

    if (ifile.startswith('/') or (ifile[1] == ':')):
        ofiles = glob.glob(ifile)
    else:
        ofiles = []
        for odir in odirs:
            ofiles.extend(glob.glob(f'{odir}/{ifile}'))

    # remove '' entries
    ofiles = [ oo for oo in ofiles if oo ]

    return ofiles


def get_dates_from_files(files, date_format):
    """
    Get dates from filenames
    
    Parameters
    ----------
    files : list
        List of filenames
    date_format : str
        Date format such as %Y-%m-%dT%H%M&S, see
        https://docs.python.org/3/library/datetime.html#format-codes

    Returns
    -------
    list
        List of datetime objects

    """
    # make regular expression pattern from date format string
    patlen = {'%C': r'\d{2}',
              '%d': r'\d{2}',
              '%g': r'\d{2}',
              '%G': r'\d{4}',
              '%H': r'\d{2}',
              '%I': r'\d{2}',
              '%j': r'\d{3}',
              '%m': r'\d{2}',
              '%M': r'\d{2}',
              '%S': r'\d{2}',
              '%u': r'\d{1}',
              '%U': r'\d{2}',
              '%V': r'\d{2}',
              '%w': r'\d{1}',
              '%W': r'\d{2}',
              '%y': r'\d{2}',
              '%Y': r'\d{4}',}
    pat = r'' + date_format
    for p in patlen:
        pat = pat.replace(p, patlen[p])
    
    if isinstance(files, str):
        # single file
        m = re.search(pat, files)
        dates = [pd.to_datetime(m.group(0), format=date_format)]
    else:
        # multiple files
        mpats = []
        for ifile in files:
            m = re.search(pat, ifile)
            mpats.append(m.group(0))
        dates = pd.to_datetime(mpats, format=date_format)

    return dates
