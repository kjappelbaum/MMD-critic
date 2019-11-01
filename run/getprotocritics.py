# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import os
import numpy as np
import click
from mmdcritic import MMDCritic


def write_outputfile(array, filename):
    """writes array to file"""
    np.save(filename, array)


@click.command("cli")
@click.argument("xpath", type=click.Path(exists=True))
@click.argument("gamma", default=0.024, type=float)
@click.argument("m", default=10, type=int)
@click.option("--kernel", type=click.Path(exists=True), default=None)
def main(xpath, gamma, m, kernel):
    print("*** starting mmdcritic ***")
    if kernel is not None:
        mmd_critic = MMDCritic.from_file(xpath, gamma, kernel)
    else:
        mmd_critic = MMDCritic.from_file(xpath, gamma)
    print(" *** getting prototypes ***")
    prototypes = mmd_critic.select_prototypes(m)
    print(" *** getting critics ***")
    critics = mmd_critic.select_criticism(m)
    write_outputfile(prototypes, "prototypes")
    write_outputfile(critics, "prototypes")


if __name__ == "__main__":
    main()
