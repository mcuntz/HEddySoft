#!/usr/bin/env python
"""
Wind direction

This module was written by Bernard Heinesch, University of Liege,
Gembloux Agro-Bio Tech, adapted by Matthias Cuntz while at Institut
National de Recherche pour l'Agriculture, l'Alimentation et
l'Environnement (INRAE), Nancy, France.

:copyright: Copyright 2025- Bernard Heinesch, Matthias Cuntz, see AUTHORS.rst for details.
:license: MIT License, see LICENSE for details.

.. moduleauthor:: Matthias Cuntz

The following functions are provided:

.. autosummary::
   wind_direction

History
   * Written, 2025, Bernard Heinesch
   * Renamed to wind_direction, Sep 2026, Matthias Cuntz
   * Lowercase sonic types, Sep 2026, Matthias Cuntz
   * Check sonic type, Sep 2026, Matthias Cuntz

"""
import numpy as np


__all__ = ['wind_direction']


def wind_direction(Usonic, Vsonic, SonicType, SonicNorthAzimuth):
    """
    see https://www.eol.ucar.edu/content/wind-direction-quick-reference
    (but there seems to be an error, with arguments of arctg2 eronously
    signed). The routine is vectorized for performance improvement here.
    Meteorological wind coordinate system: Umet, Vmet
    A positive Umet component represents wind blowing to the East.
    +Vmet is wind to the North. This is right handed with respect to an
    upward +Wmet.
    ::
        v>0
         ^
         |
         |
         +----> U>0

    Converting between Sonic and Meteorological Coordinates
    Determine the angle with respect to true north, (0=N, 90=E)
    of the +Vsonic direction.
    Call this angle Vaz. Looking from above, the sonic coordinate
    system is therefore rotated Vaz degrees clockwise from
    meteorological coordinates.

    For ATI and Campbell CSAT3 sonics, Vaz is the direction relative to
    true north, straight into the array from the un-obstructed direction,
    minus 90 degrees.
    For Gill R2 sonics, if the sign of V is flipped, then Vaz is the angle
    of the N arrow + 60. If the sign of U is flipped, then Vaz i
    s the N arrow direction + 240.
    For Gill R3s in the AXIS configuration, Vaz is the N arrow direction 
    + 240 degrees.

    From this Meteorological Coordinates, the direction the wind is
    blowing FROM is finaly computed

    Parameters
    ----------
    Usonic : float or np array of float
        sonic u wind speed component
    Vsonic : float or np array of float
        sonic v wind speed component
    SonicType : string
        type of sonic coordinate system (see respective sonic manuals)
        (either 'R2Axis','R3Spar','R3Axis','HS50Spar','HS50Axis' or 'CSAT3')
    SonicNorthAzimuth : float
        direction to which the sonic is pointing to (=0 if pointing to North)

    Returns
    -------
    Dirmet : float or np array of float
        direction the wind is coming from
        (usual meteorological convention)

    """

    # Ensure Usonic and Vsonic are arrays for vectorized operations
    Usonic = np.asarray(Usonic)
    Vsonic = np.asarray(Vsonic)

    if (SonicType.lower() == 'r2') or (SonicType.lower() == 'r2spar'):
        Usonic = -Usonic
        Vaz = 240
    elif (SonicType.lower() == 'r3spar') or (SonicType.lower() == 'hs50spar'):
        Vaz = 270
    elif (SonicType.lower() == 'r3axis') or (SonicType.lower() == 'hs50axis'):
        Vaz = 240
    elif SonicType.lower() == 'csat3':
        Vaz = 180-90
    else:
        raise ValueError(f'Sonic type not known: {SonicType}.'
                         f'Known types: R2, R2Spar, R3Spar, HS50Spar,'
                         f'R3Axis, HS50Axis, CSAT3')

    # compute wind direction in the sonic coordinate system
    Dirsonic = np.arctan2(Usonic, Vsonic) * 180./np.pi
    # adjust negative angles
    Dirsonic = np.where(Dirsonic < 0, Dirsonic + 360, Dirsonic)

    # switch to the met coordinate system
    Dirmet = Dirsonic + Vaz
    # adjust angles > 360
    Dirmet = np.where(Dirmet > 360, Dirmet - 360, Dirmet)

    # correct for SonicNorthAzimuth
    Dirmet = (Dirmet + SonicNorthAzimuth) % 360

    # switch from the "to" the "from" wind direction convention
    Dirmet = np.where(Dirmet < 180, Dirmet + 180, Dirmet - 180)

    return Dirmet
