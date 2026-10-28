#!/usr/bin/env python
"""
Read config file taking care of data types

This module was written by Bernard Heinesch, University of Liege,
Gembloux Agro-Bio Tech and modified by Matthias Cuntz while at
Institut National de Recherche pour l'Agriculture, l'Alimentation et
l'Environnement (INRAE), Nancy, France.

:copyright: Copyright 2025- Bernard Heinesch, Matthias Cuntz, see AUTHORS.rst for details.
:license: MIT License, see LICENSE for details.

.. moduleauthor:: Matthias Cuntz

The following functions are provided:

.. autosummary::
   astConfigParser
   get_config_base_dir
   read_config

History
   * Written, 2025, Bernard Heinesch
   * Rename MyConfigParser to astConfigParser, Sep 2026, Matthias Cuntz
   * Add __init__ to astConfigParser, Sep 2026, Matthias Cuntz
   * Rename get_ini to read_config, Sep 2026, Matthias Cuntz
   * Preserve keys while reading config file, Sep 2026, Matthias Cuntz
   * numpydoc format in docstrings, Sep 2026, Matthias Cuntz
   * Added get_config_base_dir, Sep 2026, Matthias Cuntz
   * Assert that basedir starts with /, Oct 2026, Matthias Cuntz

"""
import ast
import configparser


__all__ = ['astConfigParser', 'read_config', 'get_config_base_dir']


class astConfigParser(configparser.ConfigParser):
    """
    ConfigParser class where get function preserves data type in config file
    
    Parameters
    ----------
    kwargs: dict, toptional
        Keyword arguments passed to configparser.ConfigParser

    Returns
    -------
    astConfigParser
        astConfigParser object, which is an enhanced
        configparser.ConfigParser object

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get(self, section, option, *, raw=False, vars=None,
            fallback=configparser._UNSET):
        """
        Get an option value for a given section.

        The returned string of configparser.ConfigParser.get() is evaluated
        with ast.literal_eval to preserve the data type used in config file.

        If `vars` is provided, it must be a dictionary. The option is looked up
        in `vars` (if provided), `section`, and in `DEFAULTSECT` in that order.
        If the key is not found and `fallback` is provided, it is used as
        a fallback value. `None` can be provided as a `fallback` value.

        If interpolation is enabled and the optional argument `raw` is False,
        all interpolations are expanded in the return values.

        Arguments `raw`, `vars`, and `fallback` are keyword only.

        The section DEFAULT is special.

        """
        # configparser.ConfigParser.get()
        value = super().get(section, option, raw=raw, vars=vars, fallback=fallback)

        try:
            # evaluate value using ast.literal_eval
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            # if evaluation fails, return original string
            return value


def read_config(config):
    """
    Read config file taking care of data types
    
    Parameters
    ----------
    config: string
        Filename of the config file. If config is already an
        astConfigParser, it will simply be returned. 

    Returns
    -------
    astConfigParser
        astConfigParser object based on configparser.ConfigParser

    """
    if isinstance(config, astConfigParser):
        cfg = config
    elif isinstance(config, str):
        cfg = astConfigParser()
        cfg.optionxform = str  # preserve case of keys
        cfg.read(config)
    else:
        raise ValueError(
            f'Input must be string of a config filename or'
            f' astConfigParser instance that has already'
            f' read the config file. Got: {type(config)}')

    return cfg


def get_config_base_dir(config=None):
    '''
    Get base_dir from config file

    Parameters
    ----------
    config : str or astConfigParser, optional
        Config filename or heddysoft.configparser.astConfigParser
        instance of HEddySoft config file.

    Returns
    -------
    str
        absolute full path

    '''
    if config is None:
        raise ValueError('No config file given')
    
    cfg = read_config(config)
    
    base_dir = cfg.get('files', 'base_dir')
    
    if not base_dir:
        import warnings
        warnings.warn('Empty base_dir in config file')

    assert (base_dir.startswith('/') or (base_dir[1] == ':')), (
        f'base_dir must be an absolute path in config file;'
        f' given: {base_dir}')

    return base_dir
