import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy as sp

import os

import src.helper as helper

default_slice_dict = {'t': 0, 'x': 1, 'y': 2, 'z': 3, '1': 1, '2': 2, '3': 3}


def get_component(data, which='z', skip_time_steps=0):
    """
    Returns the selected component of the magnetic profile as a two-dimensional array, containing all spins of the
    chosen component for every time step. The first axis defines the time step. That means, when wanting to average
    over time, use np.average(..., axis=0).
    :param data: Data of the magnetic profile file, retrieved e.g. via np.loadtxt(file_path).
    :param which: Which component to return. For example, 't' or 'x'. Defaults to 'z'.
    :param skip_time_steps: Number of time steps to skip when returning data. Defaults to 0. Can be useful if the data takes a few time steps to equilibrate.
    :return: The selected spin component of the magnetic profile, as a function of time.
    """
    slice_dict = default_slice_dict
    if which not in slice_dict:
        raise ValueError(f"Can't return {which}-component of spin.")
    if slice_dict[which] == 0:
        return data[skip_time_steps:, slice_dict[which]]
    return data[skip_time_steps:, slice_dict[which]::3]


def get_components_as_tuple(data, which='xyz', skip_time_steps=0):
    for component in which:
        yield get_component(data, component, skip_time_steps=skip_time_steps)



# TODO: Test
def save_arrayjob_as_npy(base_path: str, npy_path: str, start: int, stop=None, step=1,
                         middle_path="spin-configs-99-999/mag-profile-99-999.altermagnet", suffix=".dat",
                         force=False):
    """
    Reads file at '{base_path}{index}/{middle_path}X{suffix}' for X=A and X=B and saves it to npy_path. Only works if all array jobs have finished or rather have produced the same amount of lines.
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

    array_job_size = int(np.ceil((1 + (stop - start)) / step))
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


# TODO: seems to be working
def save_arrayjob_as_npz(base_path: str, npz_path: str, start: int, stop=None, step=1,
                         middle_path="spin-configs-99-999/mag-profile-99-999.altermagnet", suffix=".dat",
                         force=False):
    """
    Reads file at '{base_path}{index}/{middle_path}X{suffix}' for X=A and X=B and saves it to npy_path. Only works if all array jobs have finished or rather have produced the same amount of lines.
    :param base_path:
    :param npz_path: New save path.
    :param start: Starting index or number of the array job or number of executed jobs.
    :param stop: Stopping index.
    :param step: Step size.
    :param middle_path:
    :param suffix: defaults to .dat
    :param force: If True, overwrite existing npy file.
    :return: data_dict_A, data_dict_B, npy_path
    """

    npz_path = npz_path[:-4] if npz_path[-4:] == ".npz" else npz_path
    if os.path.isfile(npz_path) and not force:
        print(f"File {npz_path} already exists, therefore nothing is saved.")
        print(f"Returning empty dictionaries instead as loading data might take very long.")
        return dict(), dict(), npz_path

    if stop is None:
        stop = start
        start = 1

    print(f"Reading from original data file in {base_path}{{i}}...")

    data_dict_A = dict()
    data_dict_B = dict()

    for i in range(start, stop + 1, step):
        print(f"{i}", end="")
        data_dict_A[str(i)] = np.loadtxt(f"{base_path}{i}/{middle_path}A{suffix}")
        print("A", end="")
        data_dict_B[str(i)] = np.loadtxt(f"{base_path}{i}/{middle_path}B{suffix}")
        print("B")

    print(f"Saving data to '{npz_path}X.npz' for X=A and X=B...")
    np.savez(f"{npz_path}A.npz", **data_dict_A)
    np.savez(f"{npz_path}B.npz", **data_dict_B)

    return data_dict_A, data_dict_B, npz_path



# TODO: seems to be working
def load_arrayjob_npyz(save_file_prefix, file_ending=".npy"):
    """

    :param save_file_prefix:
    :return:
    """
    print(f"Loading data from '{save_file_prefix}X.{file_ending}' for X=A and X=B...")
    A = np.load(f"{save_file_prefix}A.{file_ending}")
    print("A", end="")
    B = np.load(f"{save_file_prefix}B.{file_ending}")
    print("B")
    return A, B


