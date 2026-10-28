#!/usr/bin/env python
'''
Run HEddySoft between two given dates.

All parameters, dates, directories, etc. are hard-coded.

History
-------
    * Written Matthias Cuntz, Oct 2026 from run_eddypro.py

'''
import glob
import os
import shutil
import time as ptime
import numpy as np
import pandas as pd
from helper_run import copy_existing_planar_fit_timelag_file
from helper_run import get_files_and_dates
from heddysoft import heddysoft, read_config


if __name__ == '__main__':
    t1 = ptime.time()

    isdir = os.path.dirname(os.path.abspath(__file__))

    # ----------------------------------------------------------
    # Setup run: tower, startdate, enddate, only planar fit or timelag
    #

    # which tower: Eiffel (incl. scaffold) or NT (> 01.01.2013 00:00:00)
    tower = 'NT'

    # must be format: '%Y-%m-%d %H:%M:%S',
    # i.e. 'YYYY-MM-DD hh:mm:ss', e.g. '2019-02-16 03:30:00'
    # Eiffel - Start: 1997-01-01 00:00:00
    #          Suitable files from: 1998-06-29 12:30:00 or 1998-07-24 20:00:00
    #          Restart: 2000-04-01 00:00:00
    #          End: 2022-04-08 23:30:00
    # NT - Start SLT: 2012-04-23 18:00:00
    # NT - Start Smartflux: 2019-07-22 08:30:00
    startdate = '2022-06-01 18:00:00'  # first day and time
    enddate   = '2022-06-05 18:30:00'  # last day and time (not included)

    # produce timelag files only
    dotimelag = False

    # ----------------------------------------------------------
    # Immutable files and directories
    #

    # Move from Gill R2 to R3 anenometer on Eiffel tower:
    # have to set master_sonic in EddyPro config
    sdater3 = pd.to_datetime('2005-01-07 18:30:00', format='ISO8601')

    # Copy planarfit and timelag files of dates in values from dates in keys:
    # The calculation periods have different number of columns in the slt
    # files than the application period. This gives problems with EddyPro
    # metadata files. So do not redundantly recalculate the files but just
    # copy the once from the time periods before.
    sdate7 = {
        pd.to_datetime('2010-03-08 11:00:00', format='ISO8601'):
        pd.to_datetime('2009-11-15 00:00:00', format='ISO8601'),
        pd.to_datetime('2010-03-15 09:30:00', format='ISO8601'):
        pd.to_datetime('2009-11-15 00:00:00', format='ISO8601'),
        pd.to_datetime('2016-03-23 16:00:00', format='ISO8601'):
        pd.to_datetime('2015-11-15 00:00:00', format='ISO8601')}

    # Start smartflux
    sdatesmart = pd.to_datetime('2019-07-22 08:30:00', format='ISO8601')

    # Start smartflux metadata 0348: data files have 4 new columns
    sdate0348 = pd.to_datetime('2022-11-25 00:00:00', format='ISO8601')

    # base directory to which all other directories are relative to
    disk = 'J:'
    rawdir = f'{disk}/BD_Hesse/Donnees_brutes'
    calcdir = f'{disk}/calc_ec'
    dotest = False  # Do/Do not delete and copy data files

    # test @ Matthias - TimeLagHandle in file src/src_rp/timelag_handle.f90
    disk = '/Volumes/nancy-eef-prnas/Sites_ateliers'
    rawdir = f'{disk}/BD_Hesse/Donnees_brutes'
    calcdir = '/Users/cuntz/prog/github/HEddySoft/examples/FR-Hes'
    dotest = False
    # test @ Matthias

    # Input files
    # EddyPro configuration and metadata files
    configdir = f'{calcdir}/config'
    hesconfig = f'{configdir}/FR-Hes_heddysoft.cfg'
    # Extra metadata file
    ltower = tower.lower()
    # directory names
    pffile = f'{configdir}/{ltower}_planar_fit.csv'
    # Tree metadata file
    xtfile = f'{configdir}/{ltower}_height_nstem.csv'
    # Sonic and Licor metadata file
    xlfile = f'{configdir}/{ltower}_sonic_irga_metadata.csv'

    # Planar fit files
    planarfitdir = f'{calcdir}/planar_fit'
    # Meteo file base
    meteobase = f'{calcdir}/meteo/meteo_{ltower}_'

    # In/Out files
    # Timelag files
    timelagdir = f'{calcdir}/timelag'
    # Final flux files
    outputdir = f'{calcdir}/output'

    # Output files
    # working directory
    workdir = f'{calcdir}/work'
    # temporary data directory
    inputdir = f'{workdir}/input'

    # ----------------------------------------------------------
    # Clean old runs
    #

    print(f'Working directory: {workdir}')
    os.makedirs(workdir, exist_ok=True)
    print(f'Data directory: {inputdir}')
    os.makedirs(inputdir, exist_ok=True)

    # ----------------------------------------------------------
    # Make one meteo file for whole period
    #

    # s = start, e = end
    sdate = pd.to_datetime(startdate, format='ISO8601')
    edate = pd.to_datetime(enddate, format='ISO8601')
    syear = sdate.year
    eyear = edate.year
    if enddate[-14:] == '01-01 00:00:00':
        eyear -= 1
    print(f'Run datetimes {sdate} to {edate}')
    if dotimelag:
        print('Timelags only')

    mfile = f'{workdir}/meteo.csv.gz'
    if os.path.exists(mfile):
        print(f'    Remove meteo file {mfile}')
        os.remove(mfile)

    if syear < eyear:
        dfm = pd.DataFrame()
        for yy in range(syear, eyear + 1):
            print(f'        meteo year {yy}')
            idf = pd.read_csv(f'{meteobase}{yy}.csv.gz', index_col=0,
                              parse_dates=True, date_format='ISO8601')
            dfm = pd.concat([dfm, idf])
        print(f'    New meteo file {mfile}')
        dfm.to_csv(mfile, index_label=dfm.index.name)
    else:
        print(f'        meteo year {syear}')
        ifile = f'{meteobase}{syear}.csv.gz'
        print(f'    New meteo file {mfile}')
        shutil.copy2(ifile, mfile)

    if (ltower == 'nt') and (sdate >= sdatesmart):
        issmart = True
    else:
        issmart = False

    # ----------------------------------------------------------
    # Check and select dates
    #

    # The planar fit config file leads the timing.
    # The stem height and density file must have the same dates as the
    # planar fit column Apply-Start.
    # There must be planar fit files for each Apply-Start date.

    # planar fit config
    # Apply-Start,Apply-End,Calc-Start,Calc-End,Comment
    dfp = pd.read_csv(pffile)
    for cc in dfp.columns[:-1]:
        dfp[cc] = pd.to_datetime(dfp[cc], format='ISO8601')

    # take only datetimes needed
    if dotimelag:
        scol = 'Calc-Start'
        ecol = 'Calc-End'
    else:
        scol = 'Apply-Start'
        ecol = 'Apply-End'

    sidx = dfp[dfp[scol] <= sdate].iloc[-1].name
    eidx = dfp[dfp[ecol] >= edate].iloc[0].name
    dfp = dfp.iloc[sidx:eidx + 1]

    dfp['Run-Start'] = dfp[scol]
    dfp['Run-End'] = dfp[ecol]
    if not dotimelag:
        dfp.loc[dfp.index == dfp.index[0], 'Run-Start'] = sdate
        dfp.loc[dfp.index == dfp.index[-1], 'Run-End'] = edate

    print('Periods to calculate')
    print(dfp[['Run-Start', 'Run-End']])

    # tree metadata
    dfx = pd.read_csv(xtfile, index_col=0,
                      parse_dates=True, date_format='ISO8601')

    # planar fit files
    pffiles, pfdates = get_files_and_dates(
        f'{planarfitdir}/{ltower}_planar_fit_*.txt')

    # check that tree metadata dates has all dates of planar fit metadata
    for tt in dfp['Apply-Start']:
        if not any(dfx.index == tt):
            raise ValueError(f'Date {tt} of planar fit metadata {pffile}'
                             f' not in tree metadata {xtfile}')
    dfx = dfx.loc[dfp['Apply-Start']]

    # check that there is a planar fit file for each metadata entry
    for tt in dfp['Apply-Start']:
        if not any(pfdates == tt):
            raise ValueError(f'No planar fit file found for planar fit'
                             f' metadata {tt} of {pffile}')
    pffiles = [ pffiles[i] for i in range(len(pfdates))
                if any(dfp['Apply-Start'] == pfdates[i]) ]
    pfdates = [ pfdates[i] for i in range(len(pfdates))
                if any(dfp['Apply-Start'] == pfdates[i]) ]

    # sonic and irga metadata
    dfl = pd.read_csv(xlfile, sep=';', index_col=0,
                      parse_dates=True, date_format='ISO8601')
    # TODO: check that there is a planar fit entry for each change of sonic or
    # irga metadata
    for tt in dfl.index:
        if ((tt < dfp['Apply-Start'].iloc[0]) or
            (tt > dfp['Apply-Start'].iloc[-1])):
            continue
        if not (dfp['Apply-Start'] == tt).any():
            raise ValueError(f'Date {tt} of sonic and licor metadata'
                             f' {xlfile} not in planar fit metadata'
                             f' {pffile}')

    # ----------------------------------------------------------
    # Tree metadata
    #

    # d, z0 from Introduction to Micrometeorology - S. Pal Arya
    # tree canopy width (m)
    dfx['TreeWidth (m)'] = 316.227778 / np.sqrt(dfx['nStem (ha-1)'])
    # displacement height
    xwhn = dfx['TreeWidth (m)'] * dfx['CanopyHeight (m)'] * dfx['nStem (ha-1)']
    dfx['DisplacementHeight (m)'] = (dfx['CanopyHeight (m)'] *
                                     (1. - (1. - np.exp(-np.sqrt(0.52e-4 * xwhn))) /
                                      np.sqrt(0.52e-4 * xwhn)))
    # z0
    dfx['z0 (m)'] = ((dfx['CanopyHeight (m)'] - dfx['DisplacementHeight (m)']) *
                     np.exp(-0.4 / np.sqrt(0.01 + 3.5e-5 * xwhn) - 0.19314718))

    # ----------------------------------------------------------
    # Loop over periods
    #

    for pp in range(dfp.shape[0]):
        iadate = dfp['Apply-Start'].iloc[pp]
        isdate = dfp['Run-Start'].iloc[pp]
        iedate = dfp['Run-End'].iloc[pp]

        print(f'\nPeriod {pp}')
        print(f'Process {isdate} to {iedate}')

        # just copy timelag file if already calculated
        if dotimelag and (iadate in sdate7):
            copy_existing_planar_fit_timelag_file(
                'timelag', timelagdir, ltower, iadate, sdate7[iadate])
            continue

        # planar fit file
        pfile = [ pffiles[i] for i in range(len(pffiles))
                  if pfdates[i] == iadate ][0]

        # timelag file
        tfile = f'{timelagdir}/{ltower}_timelag.txt'

        # tree metadata
        dfxx = dfx[dfx.index == iadate]
        xxd = dfxx['DisplacementHeight (m)'].iloc[0]

        # Licor metadata
        idx = np.where(dfl.index <= iadate)[0]
        dfll = dfl.iloc[idx[-1]]

        # ----------------------------------------------------------
        # Mutable files and directories for this time period
        #

        ssdate = isdate.strftime('%Y%m%d%H%M%S')
        sedate = iedate.strftime('%Y%m%d%H%M%S')

        # config file
        cfile = (f'{workdir}/FR-Hes_{ltower}'
                 f'_{ssdate}_{sedate}.cfg')

        print('    Config:      ', os.path.basename(cfile))
        print('    Meteo:       ', os.path.basename(mfile))
        print('    Planar fit:  ', os.path.basename(pfile))
        print('    Timelag:     ', os.path.basename(tfile))
        if dotimelag:
            toutfile = 'FR-Hes_timelags.csv'
            print('    Timelag out: ', toutfile)

        # Copy data files into data input directory for EddyPro
        if not dotest:
            if os.path.exists(inputdir):
                shutil.rmtree(inputdir, ignore_errors=True)
            os.makedirs(inputdir)
            for yy in range(isdate.year, iedate.year + 1):
                if issmart:
                    if yy >= 2024:
                        ddir = (f'{rawdir}/{yy}/Auto/'
                                f'Smartflux{tower.upper()}/raw')
                    elif yy == 2023:
                        ddir = (f'{rawdir}/{yy}/Auto/'
                                f'Smartflux{tower.upper()}_{yy}/raw')
                    else:
                        ddir = (f'{rawdir}/{yy}/Auto/'
                                f'Smartflux{tower.upper()}_{yy}')
                else:
                    if ltower == 'nt':
                        utower = tower.upper()
                    else:
                        utower = '1'
                    ddir = f'{rawdir}/{yy}/Auto/SLT_Eddy{utower}_{yy}'
                if os.path.exists(ddir):
                    print('    Copy files year', yy)
                    if issmart:
                        afiles = glob.glob(f'{ddir}/*.ghg')
                        afiles = [ mm.replace('\\', '/') for mm in afiles ]
                        adates = pd.to_datetime(
                            [ os.path.basename(ff)[0:17] for ff in afiles ],
                            format='%Y-%m-%dT%H%M%S')
                    else:
                        afiles = glob.glob(f'{ddir}/*.[sS][lL][tT]')
                        afiles = [ mm.replace('\\', '/') for mm in afiles ]
                        adates = pd.to_datetime(
                            [ os.path.basename(ff)[1:12] for ff in afiles ],
                            format='%Y%j%H%M')
                    dfiles = [ ff for ii, ff in enumerate(afiles)
                               if ((adates[ii] >= isdate) and
                                   (adates[ii] < iedate)) ]
                    for ff in dfiles:
                        dname = inputdir + '/' + os.path.basename(ff)
                        if not os.path.exists(dname):
                            # os.symlink(ff, dname)
                            shutil.copy2(ff, dname)

        # ----------------------------------------------------------
        # Write HEddySoft config file for this period
        #

        cfg = read_config(hesconfig)

        # cfg['version']

        # cfg['site']

        cfg['dates']['start_date'] = isdate.strftime('%Y-%m-%dT%H:%M:%S')
        cfg['dates']['end_date'] = iedate.strftime('%Y-%m-%dT%H:%M:%S')

        cfg['files']['base_dir'] = calcdir
        cfg['files']['output_dir'] = outputdir
        cfg['files']['output_file'] = 'FR-Hes_output.csv'

        cfg['sonic']['dir'] = inputdir
        if issmart:
            files = '20[0-9]*-??-??T??????_*.ghg'
        else:
            files = '[a-zA-Z][0-9]*.slt'
        cfg['sonic']['files'] = files
        sonic_file_date_format = dfll['FileDateFormat']
        cfg['sonic']['file_date_format'] = sonic_file_date_format.replace('%', '%%')
        cfg['sonic']['file_ftimestep'] = str(dfll['FileFtimestep'])
        cfg['sonic']['file_type'] = dfll['FileType']
        cfg['sonic']['file_length'] = str(dfll['FileLength (s)'])
        cfg['sonic']['columns'] = dfll['SonicColumns']
        cfg['sonic']['names'] = dfll['SonicNames']
        cfg['sonic']['sonic_type'] = dfll['SonicType']
        cfg['sonic']['orientation'] = str(dfll['SonicOrientation (degreeN)'])
        cfg['sonic']['sampling_rate'] = str(dfll['SamplingRate (Hz)'])
        sonic_height = dfll['SonicHeight (m)']
        cfg['sonic']['sensor_height'] = str(sonic_height - xxd)

        cfg['irga']['files'] = files
        irga_file_date_format = dfll['FileDateFormat']
        cfg['irga']['file_date_format'] = irga_file_date_format.replace('%', '%%')
        cfg['irga']['file_ftimestep'] = str(dfll['FileFtimestep'])
        cfg['irga']['file_type'] = dfll['FileType']
        cfg['irga']['irga_type'] = dfll['IRGAType']
        cfg['irga']['columns'] = dfll['IRGAColumns']
        cfg['irga']['names'] = dfll['IRGANames']
        cfg['irga']['co2_gain'] = str(dfll['CO2Gain (ppm mV-1)'])
        cfg['irga']['co2_offset'] = str(dfll['CO2Offset (ppm)'])
        cfg['irga']['h2o_gain'] = str(dfll['H2OGain (mmol mol-1 mV-1)'])
        cfg['irga']['h2o_offset'] = str(dfll['H2OOffset (mmol mol-1)'])
        co2_type = dfll['CO2Units'].split('(')
        cfg['irga']['co2_type'] = co2_type[0].strip()
        h2o_type = dfll['H2OUnits'].split('(')
        cfg['irga']['h2o_type'] = h2o_type[0].strip()

        cfg['meteo']['dir'] = workdir
        cfg['meteo']['files'] = 'meteo.csv.gz'

        # cfg['timelag']

        # cfg['spikes']

        cfg['rotation']['mode'] = '2'
        cfg['rotation']['dir'] = planarfitdir
        cfg['rotation']['files'] = pfile

        if dotimelag:
            cfg['param']['write_spectral_files'] = 'False'

        # cfg['plot']

        with open(cfile, "w") as f:
            cfg.write(f, space_around_delimiters=False)

        # ----------------------------------------------------------
        # Run EddyPro for this period
        #

        os.chdir(workdir)

        heddysoft(cfile, dotimelag=dotimelag)


    # ----------------------------------------------------------
    # Finish
    #

    t2    = ptime.time()
    strin = ( '[m]: {:.1f}'.format((t2 - t1) / 60.)
              if (t2 - t1) > 60.
              else '[s]: {:d}'.format(int(t2 - t1)) )
    print('Time elapsed', strin)
