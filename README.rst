HEddySoft
=========

Package to calculate eddy covariance fluxes

About HEddySoft
---------------

This package calculates eddy covariance fluxes following Mauder et
al. (2013, `10.1016/j.agrformet.2012.09.006_`). It uses routines from
the package GEddySoft_ (Dumont et al. 2026,
`10.5194/essd-18-617-2026`_) of Bernard Heinesch, and got some details
from EddyPro_ (LI-COR Biosciences 2021) and EddySoft_ (Kolle and
Rebmann 2007). GEddySoft_ was initially inspired by InnFLUX_ from
Striednig et al. (2020, `10.5194/amt-13-1447-2020`_), translated and
enhanced by Bernard Heinesch.

``HEddySoft`` focuses on conventional eddy covariance fluxes of
energy, water and carbon dioxide. It also follows the so-called TERENO
strategy of Mauder et al. (2013, `10.1016/j.agrformet.2012.09.006_`)
for quality control.

Differences to GEddySoft_ are:

* Bug fixes
* Only energy, water, and CO2 fluxes
* Spike detection by Mean Absolute Deviation (MAD)
* Include plausibility limits
* Explicit setting of quality control flags
* Check for residual vertical wind
* Adapted timelag determination between sonic and IRGA signals
* Support Edisol (EdiRE, University of Edinburgh, UK) and Eddysoft slt
  files
* Conversion of wet to dry mixing ratio for closed path analyzers
* ASCII output
* Use joblib parallelization
* Provide command line application
* Possible timelag determination only


Installation
------------

``HEddySoft`` can be installed with `pip`:

.. code-block:: bash

   python -m pip install git+https://github.com/mcuntz/HEddySoft.git

For development, you clone the repository and install it editable:

.. code-block:: bash

   git clone https://github.com/mcuntz/HEddySoft.git
   python -m pip install -ve HEddySoft


Usage
-----

``HEddySoft`` is started from the command line given a config file:

.. code-block:: bash

   heddysoft example.cfg

or from within Python:

.. code-block:: python

   from heddysoft import heddysoft
   heddysoft('example.cfg')

A documented config file is given in
`examples/FR-Hes/config/FR-Hes_heddysoft.cfg`.

At the ICOS_ site FR-Hes_, we use a wrapper script `bin/run_heddysoft.py` that edits the config file and launches ``HEddySoft`` over longer time periods with different planar fit files, configurations, etc.


License
-------

``HEddySoft`` is distributed under the MIT License. See the LICENSE_ file
for details.

Copyright (c) 2026- Matthias Cuntz

GEddySoft_ by Bernard Heinesch from the Gembloux Agro-Bio Tech Faculty of the University of Liege, Belgium was published under `Apache License 2.0`_.


.. _LICENSE: https://github.com/mcuntz/HEddySoft/blob/main/LICENSE
.. _Apache License 2.0: http://www.apache.org/licenses/LICENSE-2.0
.. _EddyPro: https://www.licor.com/EddyPro
.. _EddySoft: https://www.bgc-jena.mpg.de/5362500/tech_report10.pdf
.. _FR-Hes: https://mcuntz.github.io/fastforest/
.. _GEddySoft: https://github.com/BernardHeinesch/GEddySoft
.. _ICOS: https://www.icos-cp.eu
.. _Innflux: https://git.uibk.ac.at/acinn/apc/innflux
.. _10.1016/j.agrformet.2012.09.006: http://dx.doi.org/10.1016/j.agrformet.2012.09.006
.. _10.5194/amt-13-1447-2020: https://doi.org/10.5194/amt-13-1447-2020
.. _10.5194/essd-18-617-2026: https://doi.org/10.5194/essd-18-617-2026
