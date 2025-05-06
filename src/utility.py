import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy as sp

import bulk_util
import mag_util

from bulk_util import plot_spin_xyz_over_t

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


# TODO: Test
def save_arrayjob_as_npy(base_path: str, npy_path: str, start: int, stop=None, step=1,
                         middle_path="spin-configs-99-999/mag-profile-99-999.altermagnet", suffix=".dat",
                         force=False):
    """
    Reads file at '{base_path}{index}/{middle_path}X{suffix}' for X=A and X=B and saves it to npy_path.
    :param base_path:
    :param npy_path: New save path.
    :param start: Starting index or number of the array job or number of executed jobs.
    :param stop: Stopping index.
    :param step: Step size.
    :param middle_path:
    :param suffix: defaults to .dat
    :param force: If True, overwrite existing npy file.
    :return: data_arrA, data_arrB, index_list, npy_path
    """

    npy_path = npy_path[:-4] if npy_path[-4:] == ".npy" else npy_path
    if os.path.isfile(npy_path) and not force:
        print(f"File {npy_path} already exists, therefore nothing is saved.")
        print(f"Returning empy arrays and empty index list instead as loading data might take very long.")
        return np.empty(0), np.empty(0), [], npy_path

    if stop is None:
        stop = start
        start = 1

    array_job_size = np.ceil((1 + (stop - start)) / step)
    print(f"Reading from original data file in {base_path}i...")
    data_first_A = np.loadtxt(f"{base_path}{start}/{middle_path}A{suffix}")
    data_first_B = np.loadtxt(f"{base_path}{start}/{middle_path}B{suffix}")
    if data_first_A.shape != data_first_B.shape:
        raise ValueError(f"Somehow the dimensions of {base_path}{start}/{middle_path}X{suffix} differ for X=A and X=B.")
    data_arrA = np.empty((array_job_size,) + data_first_A.shape, dtype=data_first_A.dtype)
    data_arrB = np.empty((array_job_size,) + data_first_A.shape, dtype=data_first_A.dtype)
    data_arrA[start] = data_first_A
    data_arrB[start] = data_first_B

    index_list = [start, ]

    for i in range(start + step, stop + 1, step):
        index_list.append(i)
        print(f"{i}", end="")
        data_arrA[i] = np.loadtxt(f"{base_path}{i}/{middle_path}A{suffix}")
        print("A", end="")
        data_arrB[i] = np.loadtxt(f"{base_path}{i}/{middle_path}B{suffix}")
        print("B")

    print(f"Saving data to '{npy_path}X.npy' for X=A and X=B...")
    np.save(f"{npy_path}A.npy", data_arrA)
    np.save(f"{npy_path}B.npy", data_arrB)

    return data_arrA, data_arrB, index_list, npy_path


# TODO: Test
def load_arrayjob_npy(save_file_prefix):
    """

    :param save_file_prefix:
    :return:
    """
    print(f"Loading data from '{save_file_prefix}X.npy' for X=A and X=B...")
    A = np.load(f"{save_file_prefix}A.npy")
    print("A", end="")
    B = np.load(f"{save_file_prefix}A.npy")
    print("B")
    return A, B


def time_avg(spin_data):
    return np.average(spin_data, axis=0)
