#!/usr/bin/env python3

# Copyright (C) 2021 elphmod Developers
# This program is free software under the terms of the GNU GPLv3 or later.

import setuptools

with open('README.md', 'r', encoding='utf-8') as README:
    long_description = README.read()

setuptools.setup(
    name                          = 'elphmod',
    version                       = '2021.1',
    author                        = 'elphmod Developers',
    author_email                  = '',
    description                   = 'Modules to handle electron-phonon models',
    long_description              = long_description,
    long_description_content_type = 'text/markdown',
    url                           = 'https://bitbucket.org/berges/elphmod',
    packages                      = setuptools.find_packages(),
    python_requires               = '>=2.7',
    install_requires              = ['numpy', 'scipy', 'mpi4py'],
    classifiers                   = [
        'Programming Language :: Python',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: POSIX :: Linux',
        ],
    )
