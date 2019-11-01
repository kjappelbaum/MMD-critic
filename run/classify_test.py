# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import click
import os
import numpy as np
from mmdcritic import Classifier
from mmdcritic import MMDCritic
from sklearn.preprocessing import StandardScaler


@click.command('cli')
@click.argument('xpath')
@click.argument('ypath')
@click.argument('xtest')
@click.argument('ytest')
def main(xpath, ypath, xtest, ytest):
    KERNELWIDTHS = [0.1963, 0.1, 0.0573, 0.0270, 0.01]
    NUMBERPROTOS = 500
    X = np.load(xpath)
    y = np.load(ypath)
    Xtest = np.load(xtest)
    ytest = np.load(ytest)

    scaler = StandardScaler()

    X = scaler.fit_transform(X)
    Xtest = scaler.transform(X)

    for WIDTH in KERNELWIDTHS:
        mmd_critic = MMDCritic(X, WIDTH)
        os.remove(
            'kernel.npy'
        )  # ToDo: make this less dirty, maybe move save to __init___

        print(' *** getting prototypes ***')
        prototypes = mmd_critic.select_prototypes(NUMBERPROTOS)
        prototypes = list(prototypes)

        classifier = Classifier()
        classifier.build_model(X[prototypes], y[prototypes])
        accuracy = classifier.classify(Xtest, ytest)
        print(('Accuracy {:.3f} for gamma {}'.format(accuracy, WIDTH)))


if __name__ == '__main__':
    main()  # pylint:disable=no-value-for-parameter
