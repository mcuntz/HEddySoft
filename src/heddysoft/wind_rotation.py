#!/usr/bin/env python
"""
Wind rotation

This module was written by Bernard Heinesch, University of Liege,
Gembloux Agro-Bio Tech, adapted by Matthias Cuntz while at Institut
National de Recherche pour l'Agriculture, l'Alimentation et
l'Environnement (INRAE), Nancy, France.

:copyright: Copyright 2025- Bernard Heinesch, Matthias Cuntz, see AUTHORS.rst for details.
:license: MIT License, see LICENSE for details.

.. moduleauthor:: Matthias Cuntz

The following functions are provided:

.. autosummary::
   wind_rotation

History
   * Written, March 2025, Bernard Heinesch
   * numpy to pandas.DataFrame, Sep 2026, Matthias Cuntz
   * Pass parameters instead of configParser, Sep 2026, Matthias Cuntz

"""
import numpy as np


__all__ = ['wind_rotation']


def wind_rotation(df, rotation_mode=0, wd_mean=None,
                  rotation_matrix=None):
    """
    Coordinate rotations for sonic anemometer data

    Parameters
    ----------
    df : pandas.DataFrame
        Sonic and IRGA data
    rotation_mode : int, optional
        0: no tilt correction (default),
        1: double rotation,
        2: directional planar fit
    wd_mean: array_like, optional
        Mean wind direction, needed only for angle-dependent planar fit
    rotation_matrix: dict of 3x3 nd-arrays, optional
        Needed only for angle-dependent planar fit

    Returns
    -------
    tuple
        wind_rotated: np array of floats, rotated mean wind components
        rot_phi: float, pitch angle (rad)
        R_tilt: 3x3 np array of floats, rotation matrix

    """

    if rotation_mode == 0:
        # no rotation
        wind_rotated = [df['u'], df['v'], df['w']]  # untilted u, v and w
        rot_phi = np.nan
        R_tilt = np.array([])

    elif rotation_mode == 1:
        # double rotation (yaw and pitch)
        mdf = df[['u', 'v', 'w']].mean(axis=0)
        rot_theta = np.arctan2(mdf['v'], mdf['u'])  #yaw
        u1 = mdf['u'] * np.cos(rot_theta) + mdf['v'] * np.sin(rot_theta)
        w1 = mdf['w']
        rot_phi = np.arctan2(w1, u1)  # pitch

        R_tilt = np.array([
            [np.cos(rot_theta)*np.cos(rot_phi),
             np.sin(rot_theta)*np.cos(rot_phi),
             np.sin(rot_phi)],
            [-np.sin(rot_theta),
             np.cos(rot_theta),
             0],
            [-np.cos(rot_theta)*np.sin(rot_phi),
             -np.sin(rot_theta)*np.sin(rot_phi),
             np.cos(rot_phi)] ])

        wind_rotated = np.transpose(np.dot(R_tilt,
            [np.transpose(df['u']),
             np.transpose(df['v']),
             np.transpose(df['w']) ]))  # untilted u, v and w

    elif rotation_mode == 2:
        # angle-dependent planar fit
        if wd_mean is None:
            raise ValueError('No mean wind direction given'
                             ' in case of planar fit.')
        if rotation_matrix is None:
            raise ValueError('No planar parameters given.')

        # Get the planar fit matrix (pitch and roll)
        # for the specific wind sector
        keys = list(rotation_matrix.keys())
        for i in range(len(keys)-1):
            lower_range = keys[i]
            upper_range = keys[i+1]
            if lower_range <= wd_mean < upper_range:
                R_tilt = rotation_matrix[lower_range]
                break
        # Handle the case where wd_mean is above the last key
        else:
            # If no match is found in the loop,
            # assign the last key's value
            R_tilt = rotation_matrix[keys[-1]]

        # apply pitch and roll rotations
        wind_detilted_PFM = np.transpose(np.dot(R_tilt,
            [np.transpose(df['u']),
             np.transpose(df['v']),
             np.transpose(df['w'])]))  # untilted u, v and w

        # apply yaw rotation
        mean_uvw_PFM = np.nanmean(wind_detilted_PFM[:, 0:3], 0)

        rot_theta = np.arctan2(mean_uvw_PFM[1], mean_uvw_PFM[0])

        R_tilt = np.array([
            [np.cos(rot_theta) , np.sin(rot_theta) , 0],
            [-np.sin(rot_theta), np.cos(rot_theta) , 0],
            [        0         ,         0         , 1] ])

        wind_rotated = np.transpose(np.dot(R_tilt,
            [np.transpose(wind_detilted_PFM[:, 0]),
             np.transpose(wind_detilted_PFM[:, 1]),
             np.transpose(wind_detilted_PFM[:, 2]) ]))

        # unused
        rot_phi = np.nan

    else:
        # invalid choice
        raise ValueError(f'rotation_mode unknown: {rotation_mode}')

    return wind_rotated, rot_phi, R_tilt
