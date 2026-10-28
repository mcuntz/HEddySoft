#!/usr/bin/env python
"""
Run HEddySoft

Usage
-----
usage: heddysoft [-h] [config_file]

Run HEddySoft.

positional arguments:
  config_file  Config file

options:
  -h, --help   show this help message and exit

:copyright: Copyright 2026- Matthias Cuntz, see AUTHORS.rst for details.
:license: MIT License, see LICENSE for details.

History
   * Written Oct 2026 by Matthias Cuntz (mc (at) macu (dot) de)

"""
import argparse
from .heddysoft import heddysoft


__all__ = ['main']


def main():

    # command line interface
    dotimelag = False

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Run HEddySoft.')

    hstr = ('If set, calculate timelags only, otherwise calculate fluxes.')
    parser.add_argument('-t', '--timelag', action='store_true',
                        default=dotimelag, dest='dotimelag',
                        help=hstr)

    parser.add_argument('cfgfile', nargs='?', default=[''],
                        metavar='config_file',
                        help='Config file')

    args = parser.parse_args()
    dotimelag = args.dotimelag
    cfgfile = args.cfgfile

    heddysoft(cfgfile, dotimelag=dotimelag)

    return


if __name__ == "__main__":
    main()
