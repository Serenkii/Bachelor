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


def exp_fit_func(x, A, alpha):
    r"""
    Function of form $A e^{-\alpha (x)}$
    :param x:
    :param A:
    :param alpha:
    :return:
    """
    return A * np.exp(- alpha * x)

def exp_fit_func_2(x, A, alpha, B, beta):
    r"""
    Function of form $A e^{-\alpha (x)} + B e^{-\beta (x)}$
    :param x:
    :param A:
    :param alpha:
    :param B:
    :param beta:
    :return:
    """
    return A * np.exp(- alpha * x) + B * np.exp(- beta * x)


def linear_fit_func(x, m, c):
    r"""
    Function of form $m * x + c$
    :param x:
    :param m:
    :param c:
    :return:
    """


def perform_exp_fit():
    pass

