#!/usr/bin/env python
"""
Theoretical cospectrum according to Kaimal 1972

This module was written by Ariane Faures, University of Liege,
Gembloux Agro-Bio Tech, adapted by Matthias Cuntz while at Institut
National de Recherche pour l'Agriculture, l'Alimentation et
l'Environnement (INRAE), Nancy, France.

:copyright: Copyright 2025- Ariane Faures, Matthias Cuntz, see AUTHORS.rst for details.
:license: MIT License, see LICENSE for details.

.. moduleauthor:: Matthias Cuntz

The following functions are provided:

.. autosummary::
   kaimal_cosp
   kaimal_cosp_eddypro

History
   * Written, Nov 2022, Ariane Faures
   * Renamed Kaimal to kaimal, Sep 2026, Matthias Cuntz
   * Renamed EddyPro to eddypro, Sep 2026, Matthias Cuntz

"""


__all__ = ['kaimal_cosp', 'kaimal_cosp_eddypro']


def kaimal_cosp(nf, kf, zL):
    """
    Expression of the theoretical cospectra according to Kaimal 1972

    This can be used as the cospectra of referece as it is computed for a "perfect"
    site, i.e. fully flat and homogeneous.

    Parameters
    ----------
    nf : float
        Natural frequency
    kf : float
        Normalised frequency
    zL : float
        Stability class

    Returns
    -------
    Theoretical Kaimal cospectrum as a function of stability

    """
    if zL > 0:  # Stable conditions
        kf0 = 0.23*(1 + 6.4*zL)**0.75
        kaimal_cosp=0.81*kf/kf0/(1 + 1.5*(kf/kf0)**(2.1))/nf
        # reynolds stresses
        Au = 0.124 * ((1 + 7.9 * zL)**0.75)
        Bu = 2.34 * (Au**(-1.1))
        Cospwu = kf / (nf * (Au + Bu * kf**2.1))
    else:  # Unstable conditions
        if kf <= 1:
            kaimal_cosp=11 *kf/(1 + 13.3*kf)**(7/4)/nf
        else:
            kaimal_cosp=4 *kf/(1 +3.8*kf)**(7/3)/nf

    return kaimal_cosp


def kaimal_cosp_eddypro(fnorm, zL):
    """
    Theoretical cospectra according to Kaimal 1972 as given in EddyPro
    source code (G. Fratini)

    This can be used as the cospectra of referece as it is computed for a "perfect"
    site, i.e. fully flat and homogeneous.

    Parameters
    ----------
    fnorm : float
        Normalised frequency
    zL : float
        Stability class

    Returns
    -------
    EddyPro's Kaimal cospectra as a function of stability

    """
    if zL > 0:  # Stable conditions
        Ak = 0.284 * ((1 + 6.4 * zL)**0.75)
        Bk = 2.34  * (Ak**(-1.1))
        kaimal = fnorm / (Ak + Bk * fnorm**2.1)
    else : # Unstable conditions
        if fnorm <= 0.54:
            kaimal = 12.92 * fnorm / ((1 + 26.7 * fnorm)**1.375)
        else:
            kaimal = 4.378 * fnorm / ((1 +  3.8 * fnorm)**2.4)

    return kaimal
