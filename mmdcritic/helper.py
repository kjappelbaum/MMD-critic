# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from scipy.spatial.distance import euclidean
from six.moves import range


def write_outputfile(array, filename):
    """writes array to file"""
    np.save(filename, array)


def heuristic_guess_gamma(X, iterations=5000):
    distances = []
    length = len(X)
    for _ in range(iterations):
        index0 = np.random.randint(0, length - 1)
        index1 = np.random.randint(0, length - 1)
        distances.append(euclidean(X[index0], X[index1]))

    quantile01 = np.quantile(distances, 0.1)
    quantile05 = np.quantile(distances, 0.5)
    quantile09 = np.quantile(distances, 0.9)

    print(
        (
            'the 0.1, 0.5 and 0.9 quantiles are {:.4f}, {:.4f}, {:.4f}'.format(
                1 / quantile01, 1 / quantile05, 1 / quantile09
            )
        )
    )
