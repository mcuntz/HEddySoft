#!/usr/bin/env python
"""
Makes the HEddySoft meteo file from DB1 input files concatenated
with make_annual_meteo.py.
Fill missing data with mean daily cycle for current month.

The script shifts all dates before 05/01/2016 14:30 h by one hour
because Hesse was on UTC before and the eddy data files have hence
time stamps in UTC.

Example
-------

    # 1. set fillmeandiurnal = False
    #    and run make_meteo for tower and years
    python make_meteo.py

    # 2. run make_biomet_seasonalfile.py
    python make_meteo_seasonalfile.py

    # 3. set fillmeandiurnal = True and seasonalfile
    #    seasonalfile = 'meteo_nt_seasonalfile_2017-2022.csv'
    #    and run make_meteo for tower and years again
    python make_meteo.py


History
-------
   * Written May 2021, Matthias Cuntz
   * Use whole year meteo files, Jun 2023, Matthias Cuntz
   * Treat UTC -> GMT+1, Jun 2023, Matthias Cuntz
   * Use meteo_????_DB1.csv files, Jul 2023, Matthias Cuntz
   * New seasonalfile format, Jul 2023, Matthias Cuntz
   * pd.append to pd.concat, Jul 2023, Matthias Cuntz
   * Catch all NaN in PA column if added missing column,
     Jul 2023, Matthias Cuntz
   * Change for HEddySoft, Oct 2026, Matthias Cuntz

"""
import time as ptime
import os
import datetime as dt
import numpy as np
import pandas as pd
from manual_flags import apply_manual_flags


# Interpolate only gaps <= max_gaps
# pandas are filling at least the first limit values
# PR for maxgap is stalled since 2019
# https://github.com/pandas-dev/pandas/issues/36352
def _bfill_nan(arr):
    """ Backward-fill NaNs """
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[0]),
                   mask.shape[0] - 1)
    idx = np.minimum.accumulate(idx[::-1], axis=0)[::-1]
    out = arr[idx]
    return out


def _calc_mask(arr, maxgap):
    """ Mask NaN gaps longer than `maxgap` """
    isnan = np.isnan(arr)
    cumsum = np.cumsum(isnan).astype('float')
    diff = np.zeros_like(arr)
    diff[~isnan] = np.diff(cumsum[~isnan], prepend=0)
    diff[isnan] = np.nan
    diff = _bfill_nan(diff)
    return (diff <= maxgap) | ~isnan  # <= instead of < compared to SO answer


def _fill_seasonal(df, seasonalfile, limit=5):
    # first interpolate small gaps <= limit timesteps, using
    # https://github.com/pandas-dev/pandas/issues/36352
    # do not use interpolate directly because maxgap keyword missing
    for cc in df.columns:
        if np.any(np.isnan(df[cc])):
            df[cc] = df[cc].interpolate(
                limit=limit, limit_area='inside').where(
                    _calc_mask(df[cc], limit))

    # Then fill longer gaps
    if np.any(np.isnan(df)):
        # read seasonal mean day file
        dfs = pd.read_csv(seasonalfile, sep=',', header=0)
        scolumns = list(dfs.columns)
        for cc in df.columns:
            mask = df[cc].isna()
            if np.any(mask):
                scc = scolumns.index(cc)
                ii = np.where(mask)[0]
                for i in ii:
                    jj = np.where((dfs['Month'] == df.index[i].month) &
                                  (dfs['Hour'] == df.index[i].hour) &
                                  (dfs['Minute'] == df.index[i].minute))[0]
                    if jj.size == 0:
                        raise ValueError('Problemo 3')
                    df.loc[df.index[i], cc] = dfs.iloc[jj[0], scc]

    return df


if __name__ == '__main__':

    t1 = ptime.time()

    isdir = os.path.dirname(os.path.abspath(__file__))

    # ----------------------------------------------------------
    # Setup
    #

    # which tower: eiffel (incl. scaffold) or nt (> 2012-05-24 16:00:00)
    tower = 'nt'
    years = [2022]  # list
    # eiffel
    # years = 1997 + np.arange(19, dtype=int)  # 1997-2015
    # years = 2017 + np.arange(4, dtype=int)  # 2017-2020
    # years = 1997 + np.arange(24, dtype=int)  # 1997-2020
    # nt
    # years = 2012 + np.arange(4, dtype=int)  # 2012-2015
    # years = 2017 + np.arange(6, dtype=int)  # 2017-2022
    years = 2012 + np.arange(14, dtype=int)  # 2012-2022
    fullyear = True  # True: end on 31 December
                     # False: end with youngest date in input files
    undef = -9999.   # Set NaN and fill
    fillmeandiurnal = True  # False: keep NaN of raw files
                             # True: replace NaN by mean diurnal cycle
    # seasonalfile = 'biomet_eiffel_seasonalfile_1997-2015.csv'
    # seasonalfile = 'biomet_eiffel_seasonalfile_2017-2020.csv'
    # seasonalfile = 'biomet_nt_seasonalfile_2012-2015.csv'
    seasonalfile = 'meteo_nt_seasonalfile_2012-2025.csv'

    # base directories to which all other directories are relative to
    # # @Station_calcul_Hesse
    # prnasbase = 'Z:/Sites_ateliers/BD_Hesse/DB1'
    # outdir    = 'J:/calc_ec/biomet'
    # @Matthias
    prnasbase = '/Users/cuntz/data/inrae/hesse/BD_Hesse/DB1'
    outdir    = '/Users/cuntz/prog/github/HEddySoft/examples/FR-Hes/meteo/'

    nameout = {'PA': 'Pa_1_1_1', 'RH': 'RH_1_1_1', 'TA': 'Ta_1_1_1',
               'SW': 'Rg_1_1_1', 'LW': 'Lwin_1_1_1', 'PAR': 'PPFD_1_1_1'}
    unitout = {'PA': 'Pa', 'RH': '%', 'TA': 'K',
               'SW': 'W m-2', 'LW': 'W m-2', 'PAR': 'umol m-2 s-1'}
    orderout = ['Pa_1_1_1 (Pa)', 'RH_1_1_1 (%)', 'Ta_1_1_1 (K)',
                'Rg_1_1_1 (W m-2)', 'Lwin_1_1_1 (W m-2)',
                'PPFD_1_1_1 (umol m-2 s-1)']
    nameicos = {'PA': 'PA', 'RH': 'RH', 'TA': 'TA',
                'SW': 'SW_IN', 'LW': 'LW_IN', 'PAR': 'PPFD_IN'}

    manualfile = f'{prnasbase}/Manual_Flags_Dataloggers.csv'
    if fillmeandiurnal:
        # ToDo: read two biomet files
        # apply 1st before dutzgmt1, 2nd after dutzgmt1
        if not seasonalfile:
            raise ValueError('Set seasonalfile if fillmeandiurnal')
        seasonalfile = f'{outdir}/{seasonalfile}'
        if not os.path.exists(seasonalfile):
            raise ValueError(f'seasonalfile does not exist:'
                             f' {seasonalfile}')
    else:
        seasonalfile = ''

    # eddy system was at GMT until 05.01.2016
    dutzgmt1 = np.datetime64('2016-01-05T14:30:00')
    onehour = dt.timedelta(hours=1)

    for year in years:
        # ----------------------------------------------------------
        # Files and directories
        #

        meteofile = f'{prnasbase}/{year}/meteo_{year}_DB1.csv'
        ltower = tower.lower()
        if ltower == 'nt':
            assert year >= 2012, 'No new tower NT before 2012'
        biometfile = f'{outdir}/meteo_{ltower}_{year}.csv'

        if year == 1997:
            mvar = {'RH_3_1_1': 'RH', 'TA_3_1_1': 'TA',
                    'SW_IN_3_1_1': 'SW', 'PPFD_IN_3_1_1': 'PAR'}
        elif year == 1998:
            mvar = {'RH_3_1_1': 'RH', 'TA_3_1_1': 'TA',
                    'SW_IN_3_1_1': 'SW', 'PPFD_IN_3_1_1': 'PAR'}
        elif year == 1999:
            mvar = {'RH_3_1_1': 'RH', 'TA_3_1_1': 'TA',
                    'SW_IN_3_1_1': 'SW', 'PPFD_IN_3_1_1': 'PAR'}
        elif year == 2000:  # from scaffold to Eiffel tower
            mvar = {'RH_3_1_1': 'RH', 'TA_3_1_1': 'TA',
                    'SW_IN_3_1_1': 'SW', 'PPFD_IN_3_1_1': 'PAR',
                    'RH_2_1_2': 'RH2', 'TA_2_1_2': 'TA2',
                    'SW_IN_2_1_2': 'SW2', 'PPFD_IN_2_1_2': 'PAR2'}
        elif year == 2001:  # new radiation and humidity sensors
            mvar = {'PA_2_1_1': 'PA', 'RH_2_1_2': 'RH', 'TA_2_1_2': 'TA',
                    'SW_IN_2_1_2': 'SW', 'LW_IN_2_1_1': 'LW',
                    'PPFD_IN_2_1_2': 'PAR',
                    'RH_2_1_1': 'RH2', 'TA_2_1_1': 'TA2',
                    'SW_IN_2_1_1': 'SW2'}
        elif year == 2002:
            mvar = {'PA_2_1_1': 'PA', 'RH_2_1_1': 'RH', 'TA_2_1_1': 'TA',
                    'SW_IN_2_1_1': 'SW', 'LW_IN_2_1_1': 'LW',
                    'PPFD_IN_2_1_2': 'PAR'}
        elif year == 2003:
            mvar = {'PA_2_1_1': 'PA', 'RH_2_1_1': 'RH', 'TA_2_1_1': 'TA',
                    'SW_IN_2_1_1': 'SW', 'LW_IN_2_1_1': 'LW',
                    'PPFD_IN_2_1_2': 'PAR'}
        elif year == 2004:
            mvar = {'PA_2_1_1': 'PA', 'RH_2_1_1': 'RH', 'TA_2_1_1': 'TA',
                    'SW_IN_2_1_1': 'SW', 'LW_IN_2_1_1': 'LW',
                    'PPFD_IN_2_1_2': 'PAR'}
        elif year == 2005:
            mvar = {'PA_2_1_1': 'PA', 'RH_2_1_1': 'RH', 'TA_2_1_1': 'TA',
                    'SW_IN_2_1_1': 'SW', 'LW_IN_2_1_1': 'LW',
                    'PPFD_IN_2_1_2': 'PAR'}
        elif year == 2006:
            mvar = {'PA_2_1_1': 'PA', 'RH_2_1_1': 'RH', 'TA_2_1_1': 'TA',
                    'SW_IN_2_1_1': 'SW', 'LW_IN_2_1_1': 'LW',
                    'PPFD_IN_2_1_2': 'PAR'}
        elif year == 2007:  # new PAR sensors
            mvar = {'PA_2_1_1': 'PA', 'RH_2_1_1': 'RH', 'TA_2_1_1': 'TA',
                    'SW_IN_2_1_1': 'SW', 'LW_IN_2_1_1': 'LW',
                    'PPFD_IN_2_1_2': 'PAR', 'PPFD_IN_2_1_1': 'PAR2'}
        elif year == 2008:
            mvar = {'PA_2_1_1': 'PA', 'RH_2_1_1': 'RH', 'TA_2_1_1': 'TA',
                    'SW_IN_2_1_1': 'SW', 'LW_IN_2_1_1': 'LW',
                    'PPFD_IN_2_1_1': 'PAR'}
        elif year == 2009:
            mvar = {'PA_2_1_1': 'PA', 'RH_2_1_1': 'RH', 'TA_2_1_1': 'TA',
                    'SW_IN_2_1_1': 'SW', 'LW_IN_2_1_1': 'LW',
                    'PPFD_IN_2_1_1': 'PAR'}
        elif year == 2010:
            mvar = {'PA_2_1_1': 'PA', 'RH_2_1_1': 'RH', 'TA_2_1_1': 'TA',
                    'SW_IN_2_1_1': 'SW', 'LW_IN_2_1_1': 'LW',
                    'PPFD_IN_2_1_1': 'PAR'}
        elif year == 2011:
            mvar = {'PA_2_1_1': 'PA', 'RH_2_1_1': 'RH', 'TA_2_1_1': 'TA',
                    'SW_IN_2_1_1': 'SW', 'LW_IN_2_1_1': 'LW',
                    'PPFD_IN_2_1_1': 'PAR'}
        elif year == 2012:
            if ltower == 'nt':
                mvar = {'PA_1_1_1': 'PA', 'RH_1_1_1': 'RH', 'TA_1_1_1': 'TA',
                        'SW_IN_1_1_1': 'SW', 'LW_IN_1_1_1': 'LW',
                        'PPFD_IN_1_1_1': 'PAR'}
            else:
                mvar = {'PA_2_1_1': 'PA', 'RH_2_1_1': 'RH', 'TA_2_1_1': 'TA',
                        'SW_IN_2_1_1': 'SW', 'LW_IN_2_1_1': 'LW',
                        'PPFD_IN_2_1_1': 'PAR'}
        elif year == 2013:
            if ltower == 'nt':
                mvar = {'PA_1_1_1': 'PA', 'RH_1_2_1': 'RH', 'TA_1_2_1': 'TA',
                        'SW_IN_1_1_1': 'SW', 'LW_IN_1_1_1': 'LW',
                        'PPFD_IN_1_1_1': 'PAR'}
            else:
                mvar = {'PA_2_1_1': 'PA', 'RH_2_1_1': 'RH', 'TA_2_1_1': 'TA',
                        'SW_IN_2_1_1': 'SW', 'LW_IN_2_1_1': 'LW',
                        'PPFD_IN_2_1_1': 'PAR'}
        elif year == 2014:
            if ltower == 'nt':
                mvar = {'PA_1_1_1': 'PA', 'RH_1_2_1': 'RH', 'TA_1_2_1': 'TA',
                        'SW_IN_1_1_1': 'SW', 'LW_IN_1_1_1': 'LW',
                        'PPFD_IN_1_1_1': 'PAR'}
            else:
                mvar = {'PA_2_1_1': 'PA', 'RH_2_1_1': 'RH', 'TA_2_1_1': 'TA',
                        'SW_IN_2_1_1': 'SW', 'LW_IN_2_1_1': 'LW',
                        'PPFD_IN_2_1_1': 'PAR'}
        elif year == 2015:
            if ltower == 'nt':
                mvar = {'PA_1_1_1': 'PA', 'RH_1_2_1': 'RH', 'TA_1_2_1': 'TA',
                        'SW_IN_1_1_1': 'SW', 'LW_IN_1_1_1': 'LW',
                        'PPFD_IN_1_1_1': 'PAR'}
            else:
                mvar = {'PA_2_1_1': 'PA', 'RH_2_1_1': 'RH', 'TA_2_1_1': 'TA',
                        'SW_IN_2_1_1': 'SW', 'LW_IN_2_1_1': 'LW',
                        'PPFD_IN_2_1_1': 'PAR'}
        elif year == 2016:
            if ltower == 'nt':
                mvar = {'PA_1_1_1': 'PA', 'RH_1_2_1': 'RH', 'TA_1_2_1': 'TA',
                        'SW_IN_1_1_1': 'SW', 'LW_IN_1_1_1': 'LW',
                        'PPFD_IN_1_1_1': 'PAR'}
            else:
                mvar = {'PA_2_1_1': 'PA', 'RH_2_1_1': 'RH', 'TA_2_1_1': 'TA',
                        'SW_IN_2_1_1': 'SW', 'LW_IN_2_1_1': 'LW',
                        'PPFD_IN_2_1_1': 'PAR'}
        elif year == 2017:
            if ltower == 'nt':
                mvar = {'PA_1_1_1': 'PA', 'RH_1_2_1': 'RH', 'TA_1_2_1': 'TA',
                        'SW_IN_1_1_1': 'SW', 'LW_IN_1_1_1': 'LW',
                        'PPFD_IN_1_1_1': 'PAR'}
            else:
                mvar = {'PA_2_1_1': 'PA', 'RH_2_1_1': 'RH', 'TA_2_1_1': 'TA',
                        'SW_IN_2_1_1': 'SW', 'LW_IN_2_1_1': 'LW',
                        'PPFD_IN_2_1_1': 'PAR'}
        elif year == 2018:
            if ltower == 'nt':
                mvar = {'PA_1_1_1': 'PA', 'RH_1_2_1': 'RH', 'TA_1_2_1': 'TA',
                        'SW_IN_1_1_1': 'SW', 'LW_IN_1_1_1': 'LW',
                        'PPFD_IN_1_1_1': 'PAR'}
            else:
                mvar = {'PA_2_1_1': 'PA', 'RH_2_1_1': 'RH', 'TA_2_1_1': 'TA',
                        'SW_IN_2_1_1': 'SW', 'LW_IN_2_1_1': 'LW',
                        'PPFD_IN_2_1_1': 'PAR'}
        elif year == 2019:
            if ltower == 'nt':
                mvar = {'PA_1_1_1': 'PA', 'RH_1_2_1': 'RH', 'TA_1_2_1': 'TA',
                        'SW_IN_1_1_1': 'SW', 'LW_IN_1_1_1': 'LW',
                        'PPFD_IN_1_1_1': 'PAR'}
            else:
                mvar = {'PA_2_1_1': 'PA', 'RH_2_1_1': 'RH', 'TA_2_1_1': 'TA',
                        'SW_IN_2_1_1': 'SW', 'LW_IN_2_1_1': 'LW',
                        'PPFD_IN_2_1_1': 'PAR'}
        elif year == 2020:
            if ltower == 'nt':
                mvar = {'PA_1_1_1': 'PA', 'RH_1_2_1': 'RH', 'TA_1_2_1': 'TA',
                        'SW_IN_1_1_1': 'SW', 'LW_IN_1_1_1': 'LW',
                        'PPFD_IN_1_1_1': 'PAR'}
            else:
                mvar = {'PA_2_1_1': 'PA', 'TA_2_1_1': 'TA',
                        'SW_IN_2_1_1': 'SW', 'LW_IN_2_1_1': 'LW',
                        'PPFD_IN_2_1_1': 'PAR', 'RH_1_2_1': 'RH'}
        elif year == 2021:
            if ltower == 'nt':  # eddy system from 27 m to 30 m
                mvar = {'PA_1_1_1': 'PA',
                        'RH_1_2_1': 'RH', 'TA_1_2_1': 'TA',
                        'RH_1_1_1': 'RH2', 'TA_1_1_1': 'TA2',
                        'SW_IN_1_1_1': 'SW', 'LW_IN_1_1_1': 'LW',
                        'PPFD_IN_1_1_1': 'PAR'}
            else:
                mvar = {'PA_2_1_1': 'PA', 'RH_2_1_1': 'RH', 'TA_2_1_1': 'TA',
                        'SW_IN_2_1_1': 'SW', 'LW_IN_2_1_1': 'LW',
                        'PPFD_IN_2_1_1': 'PAR'}
        elif year >= 2022:
            if ltower == 'nt':
                mvar = {'PA_1_1_1': 'PA', 'RH_1_1_1': 'RH', 'TA_1_1_1': 'TA',
                        'SW_IN_1_1_1': 'SW', 'LW_IN_1_1_1': 'LW',
                        'PPFD_IN_1_1_1': 'PAR'}
            else:
                mvar = {'PA_2_1_1': 'PA', 'RH_2_1_1': 'RH', 'TA_2_1_1': 'TA',
                        'SW_IN_2_1_1': 'SW', 'LW_IN_2_1_1': 'LW',
                        'PPFD_IN_2_1_1': 'PAR'}
        else:
            raise ValueError(f'Year not known for filenames: {year}')
        # invert keys - values
        mvarinv = { mvar[ii]: ii for ii in mvar }

        # ----------------------------------------------------------
        # Read files
        #

        print('  Read', meteofile)
        sep = ','
        form = '%Y-%m-%d %H:%M:%S'
        if pd.__version__ < '2.0':
            date_parser = lambda date: pd.to_datetime(date, format=form)
            df = pd.read_csv(meteofile, sep=sep, header=0,
                             index_col=0, parse_dates=True,
                             date_parser=date_parser)
        else:
            df = pd.read_csv(meteofile, sep=sep, header=0,
                             index_col=0, parse_dates=True, date_format=form)

        # select columns
        mm = list(mvar.keys())
        icol = [ ii for ii in list(df.columns) if ii in mm ]
        df = df[icol]

        # undef -> NaN
        if undef is not None:
            df.replace(undef, np.nan, inplace=True)

        # Remove manually flagged data
        df = apply_manual_flags(df, manualfile, verbose=False)

        # Fill in second variables, for example at change from scaffold to
        # Eiffel tower
        columns = list(df.columns)
        is2nd = { ii: mvarinv[mvar[ii] + '2']
                  for ii in columns if mvar[ii] + '2' in mvarinv }
        if len(is2nd) > 0:
            for mm in is2nd:
                mask = df[mm].isna()
                if np.any(mask):
                    df.loc[mask, mm] = df.loc[mask, is2nd[mm]]

        # Eddy system raised from 27 to 30 m in 2021
        if (ltower == 'nt') and (year == 2021):
            dinc = np.datetime64('2021-09-27T14:00:00')
            if len(is2nd) == 0:
                raise ValueError('Problemo 1')
            for mm in is2nd:
                mask = df.index > dinc
                if np.any(mask):
                    df.loc[mask, mm] = df.loc[mask, is2nd[mm]]

        # sort in given order and remove 2nd variables
        mvar = { ii: mvar[ii] for ii in mvar if not mvar[ii].endswith('2') }
        df = df[list(mvar.keys())]

        # sort times
        df.sort_index(inplace=True)

        # Add column if variable is missing
        if len(df.columns) != 6:
            print('  Add missing columns')
            for oo in nameicos:
                ioo = nameicos[oo]
                noo = True
                for cc in df.columns:
                    if cc.startswith(ioo):
                        noo = False
                if noo:
                    okey = ioo + '_1_1_1'
                    ovar = {okey: oo}
                    mvar.update(ovar)
                    df[okey] = np.nan

        # Change names and units
        head = [ f'{nameout[mvar[ii]]} ({unitout[mvar[ii]]})'
                 for ii in df.columns ]
        df.rename(columns=dict(zip(df.columns, head)), inplace=True)
        # degC -> K
        df[f"{nameout['TA']} ({unitout['TA']})"] += 273.15
        # kPa -> Pa
        if np.any(df[f"{nameout['PA']} ({unitout['PA']})"].notna()):
            if np.nanmean(df[f"{nameout['PA']} ({unitout['PA']})"]) < 10000.:
                df[f"{nameout['PA']} ({unitout['PA']})"] *= 1000.
        # 0% < RH < 100%
        df[f"{nameout['RH']} ({unitout['RH']})"] = np.clip(
            df[f"{nameout['RH']} ({unitout['RH']})"], 0., 100.)

        # make every 30 min
        df = df.resample('30min').asfreq(np.nan)

        # # interpolate small gaps < 3 hours
        # df.interpolate(method='linear', limit=5, limit_area='inside',
        #                inplace=True)

        # Shift dates: eddy system was at GMT until 05.01.2016
        # Annual meteo files are already at GMT+1 while eddy files
        # are at GMT: shift meteo to GMT for matching eddy files
        if year <= 2016:
            index1 = df.index + onehour
            df.index = df.index.where(df.index > dutzgmt1, other=index1)
            df = df[~df.index.duplicated(keep='first')]

            # Remove dates from year after due to shift of dates
            if df.index[-1].year > year:
                df = df[df.index < np.datetime64(f'{year+1}-01-01T00:30:00')]

            # Fill beginning of year
            yy0 = np.datetime64(f'{year}-01-01T00:30:00')
            if (df.index[0] > yy0):
                # add row on 01 January 00:30:00
                lrow = df.iloc[0:1].copy()
                lrow[:] = np.nan
                lrow.index = [yy0]
                df = pd.concat([lrow, df])
                # fill with nan
                df = df.resample('30min').asfreq(np.nan)

        if fullyear:
            # Fill to end of year
            yy0 = np.datetime64(f'{year + 1}-01-01T00:00:00')
            if (df.index[-1] < yy0):
                # add row on 01 January 00:00
                lrow      = df.iloc[-1:]
                lrow[:]   = np.nan
                lrow.index = [yy0]
                df = pd.concat([df, lrow])
                # fill with nan
                df = df.resample('30min').asfreq(np.nan)

        # Fill in mean daily cycle per month
        if fillmeandiurnal:
            if np.any(np.isnan(df)):
                print('  Fill with mean diurnal')
                df = _fill_seasonal(df, seasonalfile, limit=5)
                if np.any(np.isnan(df)):
                    raise ValueError('Filling is not working')

        # ----------------------------------------------------------
        # Write output file
        #

        # sort head in 'old' order
        ii = [ head.index(i) for i in orderout ]
        head = [ head[i] for i in ii ]
        # unit = [ unit[i] for i in ii ]

        # write biomet file
        print('  Write', biometfile)
        df.to_csv(biometfile, index_label='TIMESTAMP_END',
                  date_format='%Y-%m-%d %H:%M')
