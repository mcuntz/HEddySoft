#!/usr/bin/env python
'''
Read dens_H90.txt and fit to dates of planar fit

planar fit and output files are hard-coded below
for the Eiffel tower and NT. Run script twice,
once for Eiffel and once for NT.

Written Nov 2021, Matthias Cuntz
Modified
    Mar 2023, Matthias Cuntz
        - use dens_H90.txt
        - jams -> pyjams
        - Eiffel and NT
    Jun 2023, Matthias Cuntz
        - take dates from licor metadata and planar fit files
    Jun 2024, Matthias Cuntz
        - take 3rd column (ApplyStart) from planar fit file
    Jul 2024, Matthias Cuntz
        - ApplyStart moved to 1st column in planar fit file
    Oct 2026, Matthias Cuntz
        - pyjams to pandas
    Oct 2026, Matthias Cuntz
        - remove metadata file

'''
import datetime as dt
import numpy as np
import pandas as pd


def _date_interp(x, xp, fp):
    '''
    Linear interpolation between dates (xp) if date (x) is
    between 01 May and 11 November, otherwise repeat value
    (fp) of last date

    '''
    xin = np.array([ xx.toordinal() for xx in xp ])
    year = [ xx.year for xx in x ]
    may = np.array([ dt.datetime(yy, 5, 1).toordinal() for yy in year ])
    nov = np.array([ dt.datetime(yy, 11, 15).toordinal() for yy in year ])
    xout = np.array([ xx.toordinal() for xx in x ])
    fx = np.empty(len(x))
    for ix, xx in enumerate(xout):
        if xx <= xin[0]:
            fx[ix] = fp[0]
        elif xx >= xin[-1]:
            fx[ix] = fp[-1]
        else:
            if (xx > may[ix]) and (xx <= nov[ix]):
                ii = np.where(xx < xin)[0][0]
                m = (fp[ii] - fp[ii - 1]) / (xin[ii] - xin[ii - 1])
                fx[ix] = fp[ii - 1] + m * (xx - xin[ii - 1])
            else:
                fx[ix] = fx[ix - 1]
    return fx


def _piecewise_interp(x, xp, fp):
    '''
    Call _date_interp for all days between start and end dates (xp),
    then pick values at dates (x) with numpy.interp

    '''
    imin = min(min(x), min(xp))
    imax = max(max(x), max(xp))
    dd = (imax - imin).days + 1
    xa = [ imin + dt.timedelta(days=i) for i in range(dd) ]
    fa = _date_interp(xa, xp, fp)

    ix = np.array([ xx.toordinal() for xx in x ])
    xa = np.array([ xx.toordinal() for xx in xa ])
    fx = np.interp(ix, xa, fa)

    return fx


ipath = '../examples/FR-Hes/config'

inheightfile = f'{ipath}/data/dens_H90.txt'

# # Eiffel
# planarfile = f'{ipath}/eiffel_planar_fit.csv'
# outheightfile = f'{ipath}/eiffel_height_nstem.csv'

# NT
planarfile = f'{ipath}/nt_planar_fit.csv'
outheightfile = f'{ipath}/nt_height_nstem.csv'

# planar fit tiles
dfp = pd.read_csv(planarfile)
for cc in dfp.columns[:-1]:
    dfp[cc] = pd.to_datetime(dfp[cc], format='ISO8601')

# number stems per ha and canopy height
df = pd.read_csv(inheightfile)
df['TIME'] = pd.to_datetime(df['TIME'], format='ISO8601')
df.set_index('TIME', inplace=True)

# interpolate
oheight = _piecewise_interp(dfp['Apply-Start'], df.index, df['H90'].to_numpy())
onstem = _piecewise_interp(dfp['Apply-Start'], df.index, df['tree_density'].to_numpy())

# output
print(f'Write {outheightfile}')
odf = pd.DataFrame({'CanopyHeight (m)': oheight,
                    'nStem (ha-1)': onstem},
                   index=dfp['Apply-Start'])
odf.to_csv(outheightfile, float_format='%.2f', index_label='Date')
