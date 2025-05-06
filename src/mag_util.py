import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy as sp

import helper

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

