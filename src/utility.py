import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy as sp

import src.bulk_util as bulk_util
import src.mag_util as mag_util

from src.bulk_util import plot_spin_xyz_over_t

import os.path
import os

# %% Path suffixes
profile_suffix = "spin-configs-99-999/mag-profile-99-999.altermagnetA.dat"
profile_suffix_A = "spin-configs-99-999/mag-profile-99-999.altermagnetA.dat"
profile_suffix_B = "spin-configs-99-999/mag-profile-99-999.altermagnetB.dat"
config_suffix = "spin-configs-99-999/spin-config-99-999-005000.dat"


# %%

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


def perform_linear_fit():

    def linear(x, m, c):
        return m * x + c





def add_axis_break_marking(ax, position, orientation, size=12, **kwargs):
    d = .5    # size of break diagonal

    plot_kwargs = dict(marker=[(-1., -d), (1., d)], markersize=size,
                       linestyle="none", color='k', mec='k', mew=1, clip_on=False)

    orientation = str(orientation).lower()
    if orientation in ["horizontal", "h", 1]:
        kwargs['marker'] = [(d, 1.), (-d, -1.)]
    elif orientation in ["vertical", "v", 0]:
        pass
    else:
        raise ValueError(f"Unknown orientation: {orientation}")

    for key in kwargs:
        plot_kwargs[key] = kwargs[key]

    position = str(position).lower()
    if position in ["top left", "tl", "0", "left top", "lt"]:
        x, y = 0, 1
    elif position in ["top right", "tr", "1", "right top", "rt"]:
        x, y = 1, 1
    elif position in ["bottom left", "bot left", "bl", "2", "left bottom", "left bot", "lb"]:
        x, y = 0, 0
    elif position in ["bottom right", "bot right", "br", "3", "right bottom", "right bot", "rb"]:
        x, y = 1, 0
    else:
        raise ValueError(f"Unknown position: {position}")
    ax.plot([x, ], [y, ], transform=ax.transAxes, **plot_kwargs)
    return ax


def get_time_step(path):
    file_path = None
    if not path.endswith(".argv.txt"):
        files = os.listdir(path)
        for file in files:
            if file.endswith(".argv.txt"):
                file_path = os.path.join(path, file)
        if file_path is None:
            raise ValueError(f"Could not find argument file: {path}")
    else:
        file_path = path

    sim_step = None
    iter_inner = None
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            for i, arg in enumerate(parts):
                if i >= len(parts) - 1:
                    break
                if arg == "-time-step":
                    sim_step = float(parts[i + 1])
                if arg == "-iter-inner-loop":
                    iter_inner = int(parts[i + 1])

    if sim_step is None:
        raise ValueError(f"Could not find '-time-step': {file_path}")
    if iter_inner is None:
        raise ValueError(f"Could not find '-iter-inner-loop': {file_path}")

    return sim_step * iter_inner

