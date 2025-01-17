# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
import click
from mmdcritic import heuristic_guess_gamma
from sklearn.preprocessing import StandardScaler


@click.command('cli')
@click.argument('xpath', type=click.Path(exists=True))
@click.argument('iterations', default=1000, type=int)
def main(xpath, iterations):
    X = np.load(xpath)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    heuristic_guess_gamma(X, iterations)


if __name__ == '__main__':
    main()
