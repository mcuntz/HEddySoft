#!/usr/bin/env python
"""
Air properties

This module was written by Matthias Cuntz while at Institut
National de Recherche pour l'Agriculture, l'Alimentation et
l'Environnement (INRAE), Nancy, France.

:copyright: Copyright 2026- Matthias Cuntz, see AUTHORS.rst for details.
:license: MIT License, see LICENSE for details.

.. moduleauthor:: Matthias Cuntz

The following functions are provided:

.. autosummary::
   g
   Mv
   Rd
   Rgas
   Rw
   T0
   vonKarman
   cp_dry
   cp_wet
   esat
   lambda_evap
   rhocp

History
   * Written Oct 2026, Matthias Cuntz

"""
import numpy as np


__all__ = ['g', 'Mv', 'Rd', 'Rgas', 'Rw', 'T0', 'vonKarman',
           'cp_dry', 'cp_wet', 'esat', 'lambda_evap', 'rhocp']


# Constants

# Standard gravitational acceleration (m s-2)
g = 9.81  # 9.80665
# Molar mass or water vapour (kg mol-1)
Mv = 0.01802
# Gas constant for dry air [J K-1 kg-1]
Rd = 287.04
# Ideal gas constant for air [J K-1 mol-1]
Rgas = 8.314
# Gas constant for water vapour [J K-1 kg-1]
Rw = 461.5
# Conversion Kelvin to Celcius
T0 = 273.15
# von Karman constant
vonKarman = 0.4

# Functions

def cp_dry(temp):
    """
    Dry air heat capacity at constant pressure

    Parameters
    ----------
    temp : array-like
        Temperature in [K]

    Returns
    -------
    array-like
        Dry air heat capacity at constant pressure in [J kg-1 K-1]

    Notes
    -----
    Sabbatini et al. (2018).
    Eddy covariance raw data processing for CO2 and energy fluxes
    calculation at ICOS ecosystem stations.
    International Agrophysics, 32, 495–515. doi: 10.1515/intag-2017-0043

    """
    cpd = 1005. + (temp - T0 + 23.12)**2 / 3364.

    return cpd


def cp_wet(temp, rh):
    """
    Water vapour heat capacity at constant pressure

    Parameters
    ----------
    temp : array-like
        Temperature in [K]
    rh : array-like
        Relative humidity [%]

    Returns
    -------
    array-like
        Water vapour heat capacity at constant pressure in [J kg-1 K-1]

    Notes
    -----
    Sabbatini et al. (2018).
    Eddy covariance raw data processing for CO2 and energy fluxes
    calculation at ICOS ecosystem stations.
    International Agrophysics, 32, 495–515. doi: 10.1515/intag-2017-0043

    """
    cpv = ( 1859. + 0.13 * rh +
            (0.193 + 5.6e-3 * rh) * (temp - T0) +
            (1.e-3 + 5.e-5 * rh) * (temp - T0)**2 )

    return cpv


def esat(temp):
    """
    Saturated vapour pressure from EddyPro

    This formula gives same results as that in Buck (1996)
    Buck (1996), Buck Research CR-1A User's Manual, Appendix 1
        esat = 611.21 * np.exp( (18.678 - T / 234.5) * (T / (257.14 + T)) )
    where T is in Celsius.

    Parameters
    ----------
    temp : array-like
        Temperature in [K]

    Returns
    -------
    array-like
        Equilibrium water pressure at temp in [Pa]

    """
    es = np.exp(77.345 + 0.0057 * temp - 7235. / temp) / temp**8.2

    return es


def lambda_evap(temp):
    """
    Specific heat of evaporation

    This formula gives same results as
        lambda = (−0.0000614342 * T**3 + 0.00158927 * T**2 −
                  2.36418 * T + 2500.79)
    in a large range (-30 to 50 °C)

    Parameters
    ----------
    temp : array-like
        Temperature in [K]

    Returns
    -------
    array-like
        Specific heat of evaporation in [J kg-1 K-1]

    Notes
    -----
    Sabbatini et al. (2018).
    Eddy covariance raw data processing for CO2 and energy fluxes
    calculation at ICOS ecosystem stations.
    International Agrophysics, 32, 495–515. doi: 10.1515/intag-2017-0043

    """
    lmbda = (3147.5 - 2.37 * temp) * 1000.

    return lmbda
    

def rhocp(temp, rh, pres):
    """
    Air density times heat capacity at constant pressure of wet air
    from EddyPro

    Parameters
    ----------
    temp : array-like
        Temperature in [K]
    rh : array-like
        Relative humidity [%]
    pres : array-like
        Atmospheric pressure [Pa]

    Returns
    -------
    array-like
        rho_air * Cp

    """
    # Saturated water vapour partial pressure [Pa]
    es = esat(temp)
    # Ambient water vapour partial pressure [Pa]
    ea = rh / 100. * es
    # Dry air partial pressure [Pa], as P minus water vapour partial pressure
    ed =  pres - ea
    
    # Water vapour mass density [kg(H2O) m-3]
    rho_w = ea / (Rw * temp)
    # Density of dry air [kg(dry air) m-3]
    rho_d = ed / (Rd * temp)

    rho_cp = cp_wet(temp, rh) * rho_w + cp_dry(temp) * rho_d
    
    return rho_cp
