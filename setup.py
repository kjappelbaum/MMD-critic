# -*- coding: utf-8 -*-
from __future__ import absolute_import
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='mmdcritic',
    version='0.1dev',
    packages=find_packages(),
    license='MIT',
    entry_points={
        'console_scripts': [
            'getprotocritics=run.getprotocritics:main',
            'kernelguess=run.kernelguess:main',
        ]
    },
    install_requires=requirements,
    extras_require={
        'develop': ['pytest', 'pre-commit', 'black', 'prospector', 'pylint']
    },
    long_description=open('README.md').read(),
)
