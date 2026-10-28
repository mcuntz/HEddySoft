#!/usr/bin/env python
"""
Makes mean days per month from HEddySoft meteo files

Do before and after 2016 separately because does not deal with
time change in 2016

History
-------
   * Written Matthias Cuntz, May 2021
   * Use pandas to read biomet files, Jul 2023, Matthias Cuntz
   * Use pandas.groupby to make mean diurnal cycle, Jul 2023, Matthias Cuntz
   * Read biomet files of individual years, i.e. not concatenated,
     Jul 2023, Matthias Cuntz
   * Renamed to make_biomet_seasonalfile.py, Jul 2023, Matthias Cuntz
   * Change for HEddySoft, Oct 2026, Matthias Cuntz

"""
import time as ptime
import numpy as np
import pandas as pd

# ----------------------------------------------------------
# Setup
#

# directory with concatenated meteo files
# meteodir = 'J:/calc_ec/meteo'
# # @Matthias
# meteodir = '/Users/cuntz/data/inrae/hesse/BD_Hesse/Prog/windows/meteo'
meteodir = '../examples/FR-Hes/meteo'

# which tower: eiffel (incl. scaffold) or nt (> 2012-05-24 16:00:00)
tower = 'nt'
# years = [2017, 2018, 2019, 2020]  # list
# eiffel
# years = 1997 + np.arange(19, dtype=int)  # 1997-2015
# years = 2017 + np.arange(4, dtype=int)  # 2017-2020
# nt
# years = 2012 + np.arange(4, dtype=int)  # 2012-2015
# years = 2017 + np.arange(6, dtype=int)  # 2017-2022
years = 2012 + np.arange(14, dtype=int)  # 2012-2025

# ----------------------------------------------------------
# Read, mean, write
#

t1 = ptime.time()

ltower = tower.lower()

ifile = f'{meteodir}/meteo_{ltower}_{years[0]}.csv'
print('Read', ifile)
sep = ','
form = '%Y-%m-%d %H:%M'
if pd.__version__ < '2.0':
    date_parser = lambda date: pd.to_datetime(date, format=form)
    df = pd.read_csv(ifile, sep=sep,
                     header=0,
                     parse_dates=True, index_col=0,
                     date_parser=date_parser)
else:
    df = pd.read_csv(ifile, sep=sep,
                     header=0,
                     parse_dates=True, index_col=0, date_format=form)

for year in years[1:]:
    ifile = f'{meteodir}/meteo_{ltower}_{year}.csv'
    print('Read', ifile)
    if pd.__version__ < '2.0':
        df1 = pd.read_csv(ifile, sep=sep,
                          header=0,
                          parse_dates=True, index_col=0,
                          date_parser=date_parser)
    else:
        df1 = pd.read_csv(ifile, sep=sep,
                          header=0,
                          parse_dates=True, index_col=0, date_format=form)
    df, df1 = df.align(df1, axis=1)
    df = pd.concat([df, df1])

# mean diurnal cycle per day and month
dfm = df.groupby([df.index.month, df.index.hour, df.index.minute]).mean()

# write output
if len(years) == 1:
    ofile = f'{meteodir}/meteo_{ltower}_seasonalfile_{years[0]}.csv'
else:
    ofile = (f'{meteodir}/meteo_{ltower}_seasonalfile_'
             f'{years[0]}-{years[-1]}.csv')
print('Write', ofile)
dfm.to_csv(ofile, index_label=['Month', 'Hour', 'Minute'])

t2 = ptime.time()
print('Finished in [s]: {:.2f}'.format(t2 - t1))
