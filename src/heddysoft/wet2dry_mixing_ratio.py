#!/usr/bin/env python
"""
Convert wet to dry mixing ratios

This module was written by Matthias Cuntz while at Institut
National de Recherche pour l'Agriculture, l'Alimentation et
l'Environnement (INRAE), Nancy, France.

:copyright: Copyright 2026- Matthias Cuntz, see AUTHORS.rst for details.
:license: MIT License, see LICENSE for details.

.. moduleauthor:: Matthias Cuntz

The following functions are provided:

.. autosummary::
   wet2dry_mixing_ratio

History
   * Written Oct 2026, Matthias Cuntz

"""
from .config import read_config


__all__ = ['wet2dry_mixing_ratio']


def wet2dry_mixing_ratio(df, config):
    """
    Create the output data frame

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with CO2 and H2O wet mixing ratios
    config : str or astConfigParser
        Config (filename)

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with CO2 and H2O dry mixing ratios added

    """
    cfg = read_config(config)

    irga_names = [ s.strip() for s in cfg.get('irga', 'names').split(',') ]
    co2_name = ''
    h2o_name = ''
    for ii in irga_names:
        if ii.lower().startswith('co2'):
            co2_name = ii
        if ii.lower().startswith('h2o'):
            h2o_name = ii
    if ((co2_name == '') or (h2o_name == '')):
        raise ValueError(f'irga names must include names starting with CO2'
                         f' and H2O; current names: {irga_names}')

    co2_type = cfg.get('irga', 'co2_type')
    h2o_type = cfg.get('irga', 'h2o_type')

    # first H2O so that H2O_dry can be used for CO2
    # assume that H2O unit is mmol mol-1
    if ((h2o_type.lower() == 'mixing_ratio') or
        (h2o_type.lower() == 'dry_mixing_ratio')):
        df['H2O_dry'] = df[h2o_name]
    elif ((h2o_type.lower() == 'mole_fraction') or
          (h2o_type.lower() == 'wet_mixing_ratio')):
        # hwet = ea/p
        # hdry = ea/(p-ea) = ea/p * p/(p-ea) = hwet / (1 - hwet)
        df['H2O_dry'] = df[h2o_name] / (1. - df[h2o_name] * 1e-3)
    else:
        raise ValueError(f'h2o_type {h2o_type} not known or implemented')

    # CO2
    if ((co2_type.lower() == 'mixing_ratio') or
        (co2_type.lower() == 'dry_mixing_ratio')):
        df['CO2_dry'] = df[co2_name]
    elif ((co2_type.lower() == 'mole_fraction') or
          (co2_type.lower() == 'wet_mixing_ratio')):
        # cwet = c/p
        # cdry = c/(p-ea) = c/p * p/(p-ea) = cwet / (1 - hwet) = cwet * (1 + hdry)
        df['CO2_dry'] = df[co2_name] * (1. + df['H2O_dry'] * 1e-3)
    else:
        raise ValueError(f'co2_type {co2_type} not known or implemented')
    
    return df
