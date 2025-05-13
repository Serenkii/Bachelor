import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy as sp

import src.physics as physics

def get_dimensions(path):
    lengths = dict()
    dimensions = dict()
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] != '#':
                break
            if len(parts) != 4:
                continue
            if parts[2] == '=':
                if parts[1][0] == 'd':
                    dimensions[parts[1][-1]] = int(parts[-1][:-3])
                elif parts[1][0] == 'l':
                    lengths[parts[1][-1]] = int(parts[-1])

    return dimensions, lengths


def select_SL_and_component(value_arr, sublattice, spin_component):
    """

    :param value_arr: data array as for example returned by read_spin_config_dat
    :param sublattice: A or B?
    :param spin_component: x, y, or z?
    :return:
    """
    spin_dict = dict(x=0, y=1, z=2)
    sl_dict = dict(A=0, B=1)
    return value_arr[:, :, sl_dict[sublattice], spin_dict[spin_component]]


def read_spin_config_dat(path, is_tilted=True):
    """
    Reads the data in a spin configuration file and formats it into a multidimensional array. The first two components
    correspond to the x and y position of each spin. The average over all z components is already taken, as the z layers
    are independent of each other anyway. When handling the returned array, one can select a sublattice and a spin
    component via the select_SL_and_component function.
    :param path: The path of the spin configuration file.
    :param is_tilted: If the tilted configuration is used. This method has not been tested for the non-tilted setup.
    :return: A multidimensional array containing the spin components, averaged over all z layers.
    """
    if not is_tilted:
        raise NotImplementedError("Non-tilted is untested!")

    number_sublattices = 2

    dimensions, lengths = get_dimensions(path)
    divisors = dict(x=int(dimensions['x']/lengths['x']), y=int(dimensions['y']/lengths['y']), z=int(dimensions['z']/lengths['z']))

    data = np.loadtxt(path)

    data_mins = np.min(data[:, :3], axis=0)
    data[:, :3] -= data_mins

    i, j, k, sl = (
        data[:, 0].astype(int) // divisors['x'],
        data[:, 1].astype(int) // divisors['y'],
        data[:, 2].astype(int) // divisors['z'],        # z index
        data[:, 3].astype(int) - 1      # sublattice 1 or 2 (now called SL 0 and 1)
    )

    shape = (lengths['x'], lengths['y'], lengths['z'], number_sublattices, 3)
    value_grid = np.zeros(shape) + 1000     # TODO: Change to np empty and remove addition
    value_grid[j, i, k, sl, 0] = data[:, 4]         # 4=x, 5=y, 6=z
    value_grid[j, i, k, sl, 1] = data[:, 5]         # (components of spin)
    value_grid[j, i, k, sl, 2] = data[:, 6]

    value_grid_zavg = np.average(value_grid, axis=2)       # average over k (z layers)

    return value_grid_zavg


def save_spin_config_as_npy(dat_path, save_path, is_tilted=True):
    grid_data = read_spin_config_dat(dat_path, is_tilted=is_tilted)
    np.save(save_path, grid_data)


def plot_colormap(data_grid):
    X, Y = np.meshgrid(np.arange(0, data_grid.shape[0], 1, dtype=int),
                       np.arange(0, data_grid.shape[1], 1, dtype=int),
                       sparse=True, indexing='xy')
    fig, ax = plt.subplots()
    ax.pcolormesh(X, Y, data_grid)

    plt.show()


# %% Testing

path = "/data/scc/marian.gunsch/AM_tiltedX_ttmstep_7meV_2_id2/spin-configs-99-999/spin-config-99-999-010000.dat"
path = "/data/scc/marian.gunsch/AM_tiltedX_Tstep_nernst_T2/spin-configs-99-999/spin-config-99-999-005000.dat"
data = read_spin_config_dat(path)

plot_colormap(select_SL_and_component(data, "A", "z"))
