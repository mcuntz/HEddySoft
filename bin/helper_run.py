#!/usr/bin/env python
'''
Helper functions to run HEddySoft

History
-------
    * Written Matthias Cuntz, Oct 2026 from run_eddypro.py

'''
import datetime as dt
import glob
import os
import shutil
import pandas as pd


__all__ = ['copy_existing_planar_fit_timelag_file', 'get_files_and_dates',
           'raise_error']


def copy_existing_planar_fit_timelag_file(iid, idir, tower, idate, odate):
    '''
    Copy existing planar fit or timelag files with new date

    Parameters
    ----------
    iid : str
        String to include file name. Spaces will be replaced by underscores.
    idir : str
        Directory
    tower : str
        Tower identifier
    idate : datetime
        Date of input file
    odate : datetime
        Date of output file

    Returns
    -------
    copy file

    Examples
    --------
    >>> copy_existing_planar_fit_timelag_file(
    ...     'planar fit', pfdir, ltower, isdate, sdate7)

    '''
    sid = iid.replace(' ', '_')
    ipffile = (f'{idir}/{tower}_{sid}_'
               + odate.strftime('%Y%m%d%H%M%S') + '.txt')
    opffile = (f'{idir}/{tower.lower()}_{sid}_'
               + idate.strftime('%Y%m%d%H%M%S') + '.txt')
    if os.path.exists(ipffile):
        print(f'    Copy {iid} file {ipffile} to {opffile}\n')
        shutil.copy2(ipffile, opffile)
    else:
        estr = f'Could not find {iid} file: {ipffile}'
        print(estr)
        print(f'top copy to {opffile}')
        raise ValueError(estr)

    return


def get_files_and_dates(glob_str):
    '''
    Get file names and dates in filenames from filename globbing

    Parameters
    ----------
    glob_str : string
        String for filename globbing

    Returns
    -------
    files, dates : list of str, list of datetime
        list of filenames that match filename globbing,
        list of dates datetime objects of the dates at the end
        of the filename in format '%Y%m%d%H%M%S', e.g.
        'eiffel_planar_fit_19971202123000.txt'

    Notes
    -----
    Files and dates are sorted by date.

    Examples
    --------
    >>> tlfiles, tldates = get_files_and_dates(
    ...     f'../planar_fit_timelag/eiffel_timelag_*.txt')

    '''
    files  = glob.glob(glob_str)
    files  = [ mm.replace('\\', '/') for mm in files ]

    adates = [ mm.split('_')[-1].split('.')[0] for mm in files ]
    dates = pd.to_datetime(adates, format='%Y%m%d%H%M%S')

    ii = sorted(range(len(dates)), key=dates.__getitem__)
    files  = [ files[i] for i in ii ]
    dates = dates[ii]

    return files, dates


def raise_error(iid, sdate, edate, files, dates1=None, dates2=None):
    '''
    Print out dates and files and raise ValueError

    Parameters
    ----------
    iid : str
        String to include in error message
    sdate : datetime object or str
        Start date
    edate : datetime object or str
        End date
    files : list or str
        Filename(s)
    dates1 : list of datetime objects or str, optional
        First list of dates
    dates2 : list of datetime objects or str, optional
        Second list of dates

    Returns
    -------
    raises ValueError

    Examples
    --------
    >>> raise_error('Planar fit', isdate, iedate, pffiles, pfdates)

    '''
    if dates1 is None:
        estr1 = f'Did not find suitable {iid.lower()} file'
    else:
        estr1 = f'Did not find suitable {iid.lower()} file dates.'
    print(estr1)
    print(f'for startdate {sdate}')
    print(f'or enddate {edate}')

    if dates1 is not None:
        print(f'{iid} file dates1: {dates1}')

    if dates2 is not None:
        print(f'{iid} file dates2: {dates2}')

    print(f'{iid} file(s): {files}')

    raise ValueError(estr1)

    return
