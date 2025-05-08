import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy as sp

import src.bulk_util as bulk_util
import src.mag_util as mag_util

from src.bulk_util import plot_spin_xyz_over_t

import os.path

# TODO: Test
def save_file_as_npy(old: str, new: str, force=False):
    """
    Takes a path of a data file and saves it as a npy file.
    :param old:
    :param new:
    :param force: If True, overwrite existing npy file.
    :return: The read data and the path file of the npy file.
    """
    new = new if new[-4:] == ".npy" else new + ".npy"
    if os.path.isfile(new) and not force:
        print(f"File {new} already exists, therefore nothing is saved.")
        print(f"Loading existing file instead.")
        return np.load(new), new

    print(f"Reading data from {old}...")
    data = np.loadtxt(old)
    print(f"Writing data to {new}...")
    np.save(new, data)
    return data, new





def time_avg(spin_data):
    return np.average(spin_data, axis=0)
