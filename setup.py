# -*- coding: utf-8 -*-
"""GeostatTools: A geostatistical toolbox."""
from __future__ import division, absolute_import, print_function
import os
import numpy
from setuptools import setup, find_packages
from setuptools.extension import Extension

from gstools import __version__ as VERSION


DOCLINES = __doc__.split('\n')
README = open('README.md').read()

CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: End Users/Desktop',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Natural Language :: English',
    'Operating System :: Unix',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering',
    'Topic :: Utilities',
]

EXT_MODULES = []
# python setup.py build_ext --inplace --> then sphinx build

setup_kw = {
    'name': 'gstools',
    'version': VERSION,
    'maintainer': 'Lennart Schueler, Sebastian Mueller',
    'maintainer_email': "lennart.schueler@ufz.de, sebastian.mueller@ufz.de",
    'description': DOCLINES[0],
    'long_description': README,
    'long_description_content_type': 'text/markdown',
    'author': 'Lennart Schueler, Sebastian Mueller',
    'author_email': "lennart.schueler@ufz.de, sebastian.mueller@ufz.de",
    'url': 'https://github.com/LSchueler/GSTools',
    'license': 'GPL - see LICENSE',
    'classifiers': CLASSIFIERS,
    'platforms': ['Linux'],
    'include_package_data': True,
    'install_requires': [
        'numpy',
        'scipy',
        'numba'
    ],
    'packages': find_packages(exclude=['tests*', 'docs*']),
    'ext_modules': EXT_MODULES,
    'include_dirs': [numpy.get_include()],
}

setup(**setup_kw)
