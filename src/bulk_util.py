import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy as sp

import helper

default_slice_dict = {'t': 0, 'x': 3, 'y': 4, 'z': 5, '1': 3, '2': 4, '3': 5}
# TODO: FIx this, because this will always only return for sublattice A

def plot_spin_xyz_over_t(file_path, title=""):
    """
    Takes the path of a bulk file and plots all spins over time.
    :param file_path: File path of the bulk file.
    :param title: Title of the plot. Defaults to no title.
    :return: The data of the bulk file.
    """
    data = np.loadtxt(file_path)
    fig, ax = plt.subplots()
    ax.title.set_text(title)
    ax.set_xlabel("time in ps")
    ax.set_ylabel("y")
    ax.plot(data[:, 0], data[:, 3], label="sx")
    ax.plot(data[:, 0], data[:, 4], label="sy")
    ax.plot(data[:, 0], data[:, 5], label="sz")
    ax.legend()

    plt.show()

    return data


# TODO: Test and maybe adjust name
# TODO: FIx this, because this will always only return for sublattice A
def get_components(data, which='tz', skip_time_steps=0, squeeze=False):
    """
    Returns the selected axes of the bulk data as a two-dimensional array.
    :param data: Data of the bulk file, retrieved e.g. via np.loadtxt(file_path).
    :param which: Which components to return. For example, 'txyz' returns an array with four rows containing the time axis and all components. Defaults to 'tz' only returning the time axis and the z-component of the spins.
    :param skip_time_steps: Number of time steps to skip when returning data. Defaults to 0. Can be useful if the data takes a few time steps to equilibrate.
    :param squeeze: Squeeze the data before returning it. That means, an array is reduced to be one-dimensional if only one axis was selected. Defaults to False.
    :return: The selected axes of the data.
    """
    slice_list = helper.create_slice_list(which, default_slice_dict)
    if squeeze:
        return np.squeeze(data[skip_time_steps:, slice_list])
    return data[skip_time_steps:, slice_list]

# TODO: Test
# TODO: FIx this, because this will always only return for sublattice A
def get_components_as_tuple(data, which='tz', skip_time_steps=0):
    for component in which:
        yield get_components(data, component, skip_time_steps=skip_time_steps, squeeze=True)

