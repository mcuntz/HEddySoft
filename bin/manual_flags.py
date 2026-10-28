#!/usr/bin/env python3
'''
Set data to NaN for time span given in Manual_Flags file

History
-------
   * Written 2025 by Matthias Cuntz
     Routines: apply_manual_flags

'''
import datetime as dt
import numpy as np
import pandas as pd


__all__ = ['apply_manual_flags']


def apply_manual_flags(idf, mfile, verbose=False):
    '''
    Apply Manual Flags to DB1 data

    Parameters
    ----------
    idf : pandas.DataFrame
        pandas DataFrame of DB1 data
    mfile : str
        Filename of Manual Flags file
    verbose : bool, optional
        Print out manual flagging if True

    Returns
    -------
    pandas.DataFrame idf where data points with Manual Flag 2 are set to NaN

    '''
    df = idf.copy()

    if verbose:
        print(f'Read Manual Flags file {mfile}')
    dfm = pd.read_csv(mfile, comment='#')

    dfm['start'] = pd.to_datetime(dfm['start'], format='%Y-%m-%d %H:%M')
    dfm['end'] = pd.to_datetime(dfm['end'], format='%Y-%m-%d %H:%M')

    # apply up to tofay if no date given for example in 'end'
    today = dt.datetime.today()
    dfm.fillna(today, inplace=True)

    mvars = set(dfm['variable'].values)
    for ivar in mvars:
        if ivar in df.columns:
            if verbose:
                print(f'Flagging variable {ivar}')
            dfm2 = dfm[(dfm['variable'] == ivar) & (dfm['flag'] == 2)]
            for i in range(dfm2.shape[0]):
                if verbose:
                    print(f'    {dfm2['start'].iloc[i]} - {dfm2['end'].iloc[i]}')
                df[ivar] = np.where((df.index >= dfm2['start'].iloc[i]) &
                                    (df.index <= dfm2['end'].iloc[i]),
                                    np.nan, df[ivar])

    return df
