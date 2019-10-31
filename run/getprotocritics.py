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
def main(xpath, gamma, m):
    mmd_critic = MMDCritic.from_file(xpath, gamma)
    prototypes = mmd_critic.select_prototypes(m)
    critics = mmd_critic.select_critics(m)
    write_outputfile(prototypes, "prototypes")
    write_outputfile(critics, "prototypes")


if __name__ == "__main__":
    main()
