import os

import warnings

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import scipy as sp
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter
from scipy.ndimage import median_filter

import src.physics as physics
import src.helper as helper


# %%
default_force_overwrite = False
if default_force_overwrite:
    warnings.warn("FORCE OVERWRITE ENABLED!")

# %% Functions


def infer_config_path(path):
    folder_list = path.split("/")
    if folder_list[-1] == "":       # path ends with '/'
        folder_list.pop()
        path = path[:-1]    # remove /

    return_path = path

    if path.endswith(".dat"):
        pass

    elif folder_list[-1].startswith("spin-config-99-999") and folder_list[-2] == "spin-configs-99-999":
        return_path = f"{path}.dat"

    elif folder_list[-1] == "spin-configs-99-999":
        return_path = f"{path}/spin-config-99-999-005000.dat"

    else:
        return_path = f"{path}/spin-configs-99-999/spin-config-99-999-005000.dat"

    if not os.path.exists(return_path):
        raise FileNotFoundError(f"Unable to infer data file from '{path}'. Unsuccessful attempt yielded '{return_path}'."
                                f"This file does not exist.")

    return return_path



def get_dimensions(path):
    """
    Reads the specified path
    :param path: The path to a spin-config file. (Probably other files of the same simulation work too)
    :return: Returns the dimension and actual length of the simulation as two dictionaries with keys 'x', 'y' and 'z'.
    """
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
    Selects the specified sublattice and spin component and returns a three-dimensional array whose indices specify the
    position in space.
    :param value_arr: data array as for example returned by read_spin_config_dat
    :param sublattice: A or B?
    :param spin_component: x, y, or z?
    :return: A three-dimensional arrays of shape (l.x, l.y, l.z), where l is the (actual) length of the material.
    """
    spin_dict = dict(x=0, y=1, z=2)
    sl_dict = dict(A=0, B=1)
    return value_arr[:, :, :, sl_dict[sublattice], spin_dict[spin_component]]


def read_spin_config_dat_raw(path, empty_filler=np.nan, fixed_version=True):
    if not fixed_version:
        raise NotImplementedError("Only the 'fixed' version is implemented. By that, I mean with different order of "
                                  "x, y, z")

    path = infer_config_path(path)

    number_sublattices = 2

    data = np.loadtxt(path)

    # data[1, -1] = 5

    i, j, k, sl = (
        data[:, 0].astype(int),
        data[:, 1].astype(int),
        data[:, 2].astype(int),
        data[:, 3].astype(int) - 1  # sublattice 1 or 2 (now called SL 0 and 1)
    )

    shape = (np.max(i) + 1, np.max(j) + 1, np.max(k) + 1, number_sublattices, 3)

    value_grid = np.zeros(shape)
    value_grid[:] = empty_filler
    value_grid[i, j, k, sl, 0] = data[:, 4]  # 4=x, 5=y, 6=z
    value_grid[i, j, k, sl, 1] = data[:, 5]  # (components of spin)
    value_grid[i, j, k, sl, 2] = data[:, 6]  # j i k instead of i j k because indices are weird... TODO: idk about that

    return value_grid


def read_spin_config_dat(path, is_tilted=True, shift=None, fixed_version=False, empty_value=np.nan):
    """
    Reads the data in a spin configuration file and formats it into a multidimensional array. The first two components
    correspond to the x and y position of each spin. The average over all z components is already taken, as the z layers
    are independent of each other anyway. IS THE AVERAGE TAKEN THO???? I DONT THINK SO!!!!!
    When handling the returned array, one can select a sublattice and a spin
    component via the select_SL_and_component function.
    :param empty_value:
    :param shift:
    :param path: The path of the spin configuration file.
    :param is_tilted: If the tilted configuration is used. This method has not been tested for the non-tilted setup.
    :param fixed_version: If a new definition for indices shall be used
    :return: A multidimensional array containing the spin components, averaged over all z layers.
    """

    path = infer_config_path(path)

    number_sublattices = 2

    dimensions, lengths = get_dimensions(path)
    divisors = dict(x=int(dimensions['x'] / lengths['x']), y=int(dimensions['y'] / lengths['y']),
                    z=int(dimensions['z'] / lengths['z']))

    data = np.loadtxt(path)

    data_mins = np.min(data[:, :3], axis=0)
    data[:, :3] -= data_mins

    i, j, k, sl = (
        data[:, 0].astype(int) // divisors['x'],
        data[:, 1].astype(int) // divisors['y'],
        data[:, 2].astype(int) // divisors['z'],  # z index
        data[:, 3].astype(int) - 1  # sublattice 1 or 2 (now called SL 0 and 1)
    )

    if fixed_version:
        shape = (lengths['x'], lengths['y'], lengths['z'], number_sublattices, 3)

    else:
        shape = (lengths['y'], lengths['x'], lengths['z'], number_sublattices, 3)
        i, j = j, i     # TODO: I don't know about my past decisions...

    value_grid = np.zeros(shape)
    value_grid[:] = empty_value
    value_grid[i, j, k, sl, 0] = data[:, 4]  # 4=x, 5=y, 6=z
    value_grid[i, j, k, sl, 1] = data[:, 5]  # (components of spin)
    value_grid[i, j, k, sl, 2] = data[:, 6]  # j i k instead of i j k because indices are weird... TODO: idk about that

    if not is_tilted:
        if shift is None or shift.lower()=="none":
            pass
        elif shift.lower()=="left":
            raise NotImplementedError("Shift not implemented.")
        elif shift.lower()=="right":
            raise NotImplementedError("Shift not implemented.")
        else:
            raise ValueError(f"Shift {shift} not recognized. Options: none, left, right")

    if is_tilted or shift is not None:
        if np.isnan(value_grid).any():
            raise ValueError(f"A value of the value_grid containing the data of the specified path '{path}' could not be "
                             f"set using the available data in the spin configuration file. Check the file (and maybe"
                             f"also this function).")
    else:
        print("Warning! The returned value_grid contains nan-data because of the specific lattice structure.")

    return value_grid


def npy_file(dat_path: str, npy_path=None, force=default_force_overwrite, return_data=True, is_tilted=True, shift=None,
             empty_value=np.nan):

    base_folder = "/data/scc/marian.gunsch/npy/configs/"

    data_path = infer_config_path(dat_path)

    print(f"config-path: {data_path}")

    if not npy_path:
        folder_list = data_path.split("/")
        index0 = folder_list.index("marian.gunsch")
        save_name = f"{folder_list[index0 + 1].zfill(2)}"
        for i in range(index0 + 2, index0 + 4):
            if folder_list[i] == "spin-configs-99-999":
                break
            save_name += f"_{folder_list[i]}"
        save_name += f".conf.npy"
        npy_path = f"{base_folder}{save_name}"

    data = None
    print(f"Handling config file...")
    if os.path.isfile(npy_path) and not force:
        print(f"File {npy_path} already exists. Nothing will be overwritten.")
        warnings.warn("Careful! This means that the empty_value cannot be updated!")
    else:
        print(f"Reading data from {dat_path}...")
        data = read_spin_config_dat(dat_path, is_tilted, shift, True, empty_value)
        print(f"Saving data to {npy_path}...")
        np.save(npy_path, data)

    if not return_data:
        return None

    if data is None:
        print(f"Loading data from {npy_path}...")
        data = np.load(npy_path)

    print("\n")

    return data



def npy_file_from_dict(path_dict, is_tilted=None, force=default_force_overwrite, shift=None, empty_value=np.nan):
    if not is_tilted:
        is_tilted = dict()
        for d in ["100", "010", "-100", "0-10"]:
            is_tilted[d] = False
        for d in ["110", "-110", "-1-10", "1-10"]:
            is_tilted[d] = True

    data_dict = dict()

    for key in path_dict:
        data_dict[key] = npy_file(path_dict[key], None, force, True, is_tilted[key], shift, empty_value)

    return data_dict

# %%

def average_aligned_data(data, method="default", include_edges=False, return_space_arr=False):
    if method != "default":
        raise NotImplementedError()

    from scipy.signal import convolve2d

    kernel = np.ones((2, 2))
    # sum of values in each window
    sums = convolve2d(data, kernel, mode='full')
    # how many elements contributed (important at edges)
    counts = convolve2d(np.ones_like(data), kernel, mode='full')
    averages = sums / counts

    if method == "default":
        start_centered = 1
        max_x = averages.shape[0] - start_centered
        max_y = averages.shape[1] - start_centered
        data_centered = averages[start_centered:max_x:2, start_centered:max_y:2]
        start_shifted = 0 if include_edges else 2
        max_x = averages.shape[0] - start_shifted
        max_y = averages.shape[1] - start_shifted
        data_shifted = averages[start_shifted:max_x:2, start_shifted:max_y:2]
        if return_space_arr:
            x_centered = np.linspace(start_centered - 0.5, start_centered + 2 * (data_centered.shape[0] - 1) - 0.5, data_centered.shape[0])
            y_centered = np.linspace(start_centered - 0.5, start_centered + 2 * (data_centered.shape[1] - 1) - 0.5, data_centered.shape[1])
            # x_centered = np.arange(start_centered - 0.5, 2 * data_centered.shape[0], step=2)
            # y_centered = np.arange(start_centered - 0.5, 2 * data_centered.shape[1], step=2)
            x_shifted = np.linspace(start_shifted - 0.5, start_shifted + 2 * (data_shifted.shape[0] - 1) - 0.5, data_shifted.shape[0])
            y_shifted = np.linspace(start_shifted - 0.5, start_shifted + 2 * (data_shifted.shape[1] - 1) - 0.5, data_shifted.shape[1])
            # x_shifted = np.arange(start_shifted - 0.5, 2 * (data_shifted.shape[0] - start_shifted), step=2)
            # y_shifted = np.arange(start_shifted - 0.5, 2 * (data_shifted.shape[1] - start_shifted), step=2)
            return x_centered, y_centered, data_centered, x_shifted, y_shifted, data_shifted
        return data_centered, data_shifted



def plot_colormap(data_grid, title="", rel_step_pos=0.49, show_step=False, zoom=False, save_path=None, fig_comment=None,
                  fixed_version=False, colorbar_min=None, colorbar_max=None):
    """
    Creates a 2d color-plot for a given data grid.
    :param colorbar_max:
    :param colorbar_min:
    :param data_grid: The data which will be plotted in a 2d color-plot.
    :param title: The title of the figure.
    :param rel_step_pos: The relative position of the temperature step. Is only used if show_step or zoom is True.
    :param show_step: Displays a vertical dashed line at the position of the temperature step.
    :param zoom: If True, only the data around the temperature step will be shown.
    :param save_path: The path, in which the figure will be saved.
    :param fig_comment: If specified, will add a text on the figure displaying this string.
    :param fixed_version: Does not work with show step and zoom
    :return:
    """
    data_grid = np.squeeze(data_grid)
    if not fixed_version:
        X, Y = np.meshgrid(np.arange(0, data_grid.shape[1], 1, dtype=int),
                           np.arange(0, data_grid.shape[0], 1, dtype=int),
                           sparse=True, indexing='xy')
    else:
        X, Y = np.meshgrid(np.arange(0, data_grid.shape[0], 1, dtype=int),
                           np.arange(0, data_grid.shape[1], 1, dtype=int),
                           sparse=True, indexing='xy')

    fig, ax = plt.subplots()
    if zoom:
        ax.set_aspect('auto', 'box')
    else:
        ax.set_aspect('equal', 'box')
    ax.set_title(title)
    if not fixed_version:
        im = ax.pcolormesh(X, Y, data_grid, norm=colors.CenteredNorm(), cmap='RdBu_r')
        if colorbar_min and colorbar_max:
            im = ax.pcolormesh(X, Y, data_grid, cmap='RdBu_r',
                               vmin=colorbar_min, vmax=colorbar_max)
    else:
        im = ax.pcolormesh(X, Y, data_grid.T, norm=colors.CenteredNorm(), cmap='RdBu_r')
        if colorbar_min and colorbar_max:
            im = ax.pcolormesh(X, Y, data_grid.T, cmap='RdBu_r',
                               vmin=colorbar_min, vmax=colorbar_max)
        # Note that the column index corresponds to the x-coordinate, and the row index corresponds to y.
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pcolormesh.html#axes-pcolormesh-grid-orientation
        # ahhh fuck numpy, why is this so confusing, why array[z,y,x] :( :( :(
        # I should have been consistent and do something like: data_grid[component, sl, z, y, x] TODO!
    fig.colorbar(im, ax=ax)

    if show_step:
        step_pos = helper.get_absolute_T_step_index(rel_step_pos, data_grid.shape[1])
        ax.vlines(step_pos, 0, data_grid.shape[0], colors='grey', linestyle='dashed', alpha=0.7)

    ax.margins(x=0, y=0)

    if zoom:
        step_pos = helper.get_absolute_T_step_index(rel_step_pos, data_grid.shape[1])
        ax.set_xlim(step_pos - 25, step_pos + 25)

    fig.tight_layout()

    if fig_comment:
        fig.text(0.5, 0.0, fig_comment, ha='center', va='bottom', color="green", size=5.0)

    if save_path:
        print(f"Saving to {save_path}...")
        fig.savefig(save_path)

    plt.show()



def convolute(data, filter="denoise", denoise_kwargs=None):
    """
    Convolute the given two-dimensional data array with one of a few filters. A three-dimensional array is also allowed,
    as long as the array is squeezable into a two-dimensional array.
    The implementation of this method can be improved.
    Available filters:
    'none' returns a copy of the original array.
    'uniform' applies the scipy uniform filter.
    'denoise' applies the denoise_bilateral filter from skimage.restoration.
    :param data:
    :param filter: The filter that should be applied: 'none', 'uniform' or 'denoise'
    :param denoise_kwargs: If the specified filter is 'denoise', one can supply additional keyword arguments.
    :return: A copy of the original array with the filter applied.
    """
    copy = np.squeeze(data)

    if filter is None or filter == 0 or filter == "none" or filter == "copy":
        return copy

    elif filter == "uniform" or filter == "uni" or filter == 1:
        return uniform_filter(copy, size=4)

    elif filter == "denoise" or filter == 2:
        from skimage.restoration import denoise_bilateral
        if denoise_kwargs:
            return denoise_bilateral(copy, **denoise_kwargs)
        return denoise_bilateral(copy, sigma_color=0.001, sigma_spatial=4, mode='edge')  # TODO: Finetune even further

    raise ValueError(f"Unknown filter '{filter}'.")


def average_z_layers(data, *args, force_return_tuple=False, keepdims=True):
    """
    Takes a minimum of one data arrays of spin configurations and returns them averaged along z direction for each.
    :param data: A spin configuration array.
    :param args: (optional) additional spin configuration arrays.
    :param force_return_tuple: If True, also returns a tuple if only one spin configuration array was passed.
    :return: Returns either a single data array averaged along z direction, or a tuple of all passed data arrays.
    """
    if len(args) == 0 and not force_return_tuple:
        return np.mean(data, axis=2, keepdims=keepdims)

    data_tuple = (data,) + args
    return_tuple = ()
    for arg in data_tuple:
        return_tuple = return_tuple + (np.mean(arg, axis=2, keepdims=keepdims),)
    return return_tuple


# UNTESTED
def create_profile(grid_data, profile_direction, avg_slice=slice(None), which="xyz", force_return_dict=False,
                   fixed_version=True):
    """
    If the grid_data is aligned data, it is important to empty_value=np.nan when reading data. Otherwise, the average
    is wrong!
    :param grid_data:
    :param profile_direction: profile along x or y direction?
    :param avg_slice: Slice which is applied in the other direction.
    :param which: Components
    :param force_return_dict: Always return a dictionary, even when which only consists of one component
    :param fixed_version:
    :return:
    """
    warnings.warn("Untested!")

    if not fixed_version:
        warnings.warn("Use fixed version!", DeprecationWarning)
        raise NotImplementedError("Only the fixed version is implemented.")

    data = np.copy(grid_data)
    if profile_direction == "y":
        permutation = (1, 0)        # transform into case "x":
    elif profile_direction == "x":
        permutation = (0, 1)        # default case
    else:
        raise ValueError("Profiles possible only along x and y direction!")

    data = np.transpose(data, permutation + tuple(range(2, len(data.shape))))

    data = data[:, avg_slice]

    profiles = dict()

    for SL in ["A", "B"]:
        profiles[SL] = dict()
        for component in which:
            S = select_SL_and_component(data, SL, component)
            profiles[SL][component] = np.nanmean(S, axis=(1, 2))

    if len(which) == 1 and not force_return_dict:
        return profiles["A"][which], profiles["B"][which]
    return profiles["A"], profiles["B"]


# UNTESTED!
def spin_current(grid_data, current_direction, cryst_direction, profile_return=None, do_unit_cell_avg=True, normed_units=False):
    """

    :param grid_data:
    :param current_direction: Calculate spin in 'x' or 'y' direction? If None, config data is returned.
    :param cryst_direction: Determines sign and mean of calculation
    :param profile_return: 'x', 'y' or None if config data shall be returned.
    :param do_unit_cell_avg: If this is True, appropriate averages (convolutions) are taken in the aligned case. The tilted case is always in terms of unit cells.
    :return:
    """
    if not profile_return in ["x", "y", None]:
        raise ValueError("'profile_return' must be 'x' or 'y' or None to return all the data")
    if not current_direction in ["x", "y"]:
        raise ValueError("'data_direction' must be 'x' or 'y'.")
    if cryst_direction in ["-100", "-1-10", "0-10", "1-10"]:
        warnings.warn("Spin current along a direction which is opposite of the available profile directions.")
    elif not cryst_direction in ["100", "010", "110", "-110"]:
        raise ValueError("Invalid 'cryst_direction'.")

    data = np.copy(grid_data)
    if current_direction == "y":
        permutation = (1, 0)        # transform into case "x":
    elif current_direction == "x":
        permutation = (0, 1)        # default case
    else:
        raise ValueError("'data_direction' must be 'x' or 'y'.")

    data = np.transpose(data, permutation + tuple(range(2, len(data.shape))))


    tilted = False if cryst_direction in ["100", "010", "-100", "0-10"] else True
    sign = +1 if cryst_direction in ["100", "010", "110", "-110"] else -1


    def handle_tilted():
        Ja = physics.J2b
        Jb = physics.J2a
        if cryst_direction in ["-110", "1-10"]:
            Ja, Jb = Jb, Ja

        se = select_SL_and_component
        curr_au = - (Ja *
                    (se(data, "A", "x")[:-1] * se(data, "A", "y")[1:] -
                     se(data, "A", "y")[:-1] * se(data, "A", "x")[1:])
                    + Jb *
                    (se(data, "B", "x")[:-1] * se(data, "B", "y")[1:] -
                     se(data, "B", "y")[:-1] * se(data, "B", "x")[1:])
                    )

        assert curr_au.shape[0] == data.shape[0] - 1

        return curr_au


    def handle_aligned():
        J1 = physics.J1
        se = select_SL_and_component
        # row 0, 2, 4, ... --> 0::2
        curr1a = - J1 * (se(data, "B", "x")[:-1:2, 0::2] * se(data, "A", "y")[1::2, 0::2] -
                         se(data, "B", "y")[:-1:2, 0::2] * se(data, "A", "x")[1::2, 0::2])
        curr1b = - J1 * (se(data, "A", "x")[1:-2:2, 0::2] * se(data, "B", "y")[2:-1:2, 0::2] -
                         se(data, "A", "y")[1:-2:2, 0::2] * se(data, "B", "x")[2:-1:2, 0::2])

        shape1a = curr1a.shape
        shape1b = curr1b.shape
        assert shape1a[1:] == shape1b[1:]       # These checks are valid because we know the grid dimensions
        assert shape1a[0] == shape1b[0] + 1             #  are even numbers
        shape1 = (shape1a[0] + shape1b[0], ) + shape1a[1:]

        assert shape1[0] % 2 == 1   # This must now be odd because we have lost one column due to method of calculating
        assert shape1[0] + 1 == data.shape[0]
        curr1 = np.empty(shape1, dtype=float)
        curr1[0::2] = curr1a        # interleave both
        curr1[1::2] = curr1b

        # row 1, 3, 5, ... --> 1::2
        curr2a = - J1 * (se(data, "A", "x")[:-1:2, 1::2] * se(data, "B", "y")[1::2, 1::2] -
                         se(data, "A", "y")[:-1:2, 1::2] * se(data, "B", "x")[1::2, 1::2])
        curr2b = - J1 * (se(data, "B", "x")[1:-2:2, 1::2] * se(data, "A", "y")[2:-1:2, 1::2] -
                         se(data, "B", "y")[1:-2:2, 1::2] * se(data, "A", "x")[2:-1:2, 1::2])

        shape2a = curr2a.shape
        shape2b = curr2b.shape
        assert shape2a[1:] == shape2b[1:]
        assert shape2a[0] == shape2b[0] + 1
        shape2 = (shape2a[0] + shape2b[0], ) + shape2a[1:]

        assert shape2[0] % 2 == 1
        assert shape2[0] + 1 == data.shape[0]
        curr2 = np.empty(shape2, dtype=float)
        curr2[0::2] = curr2a
        curr2[1::2] = curr2b

        # Put different rows together
        assert shape1 == shape2     # This must be the same because of even grid dimensions
        curr_shape = (shape1[0], ) + (shape1[1] + shape2[1], ) + shape1[2:]
        curr = np.empty(curr_shape, dtype=float)
        curr[:, 0::2] = curr1
        curr[:, 1::2] = curr2

        assert curr.shape[0] + 1 == data.shape[0]
        assert curr.shape[1:3] == data.shape[1:3]

        return curr


    if tilted:
        current = handle_tilted() * physics.handle_spin_current_unit_prefactor(tilted, normed_units)
    else:
        current = handle_aligned() * physics.handle_spin_current_unit_prefactor(tilted, normed_units)

    # We need to multiply times two, to get the area of a unit cells --> make it comparable to the tilted variant,
    # in terms of the area the current flows through. (~ two 'atom points')
    if not tilted:
        current *= 2

    current *= sign

    if profile_return:
        current_ = np.transpose(current, permutation + (2, ))

        if profile_return == "x":
            avg_axes = (1, 2)
        elif profile_return == "y":
            avg_axes = (0, 2)
        else:
            raise ValueError("'profile_return' must be 'x' or 'y' or None to return all the data")

        return np.mean(current_, axis=avg_axes)

    if not do_unit_cell_avg:
        if not tilted:
            print("The current was multiplied by 2 to correct for the dimensionless area.")
        if tilted:
            print("For the tilted setup, the unit cell averaging is always done!")
        return np.transpose(np.mean(current, axis=2), permutation)

    raise NotImplementedError("The unit cell average has not been implemented yet.")





# %% Old and unwanted (just keep for backwards compatibility)


def save_spin_config_as_npy(dat_path, save_path, is_tilted=True):
    """
    Saves the spin configuration file in the specified data path as a new .npy file in the specified path. Any existing
    file will be overwritten.
    :param dat_path: The path of the spin configuration file.
    :param save_path: The npy file path.
    :param is_tilted: Specify if the tilted configuration is used.
    :return:
    """
    grid_data = read_spin_config_dat(dat_path, is_tilted=is_tilted)
    np.save(save_path, grid_data)


# TODO: Fix
def read_spin_config_arrjob(path_prefix, path_suffix, start, stop=None, step=1, average=True, is_tilted=True):
    if stop is None:
        stop = start
        start = 1

    array_job_size = int(np.ceil((1 + (stop - start)) / step))

    data_first = read_spin_config_dat(f"{path_prefix}{start}{path_suffix}", is_tilted=is_tilted)

    data_arr = np.empty((array_job_size,) + data_first.shape, dtype=data_first.dtype)
    data_arr[0] = data_first

    for i in range(1, array_job_size):
        job_index = start + i * step
        print(f"{job_index}.", end="")
        data_arr[i] = read_spin_config_dat(f"{path_prefix}{job_index}{path_suffix}", is_tilted=is_tilted)

    if average:
        return np.mean(data_arr, axis=0)
    return data_arr


# TODO: Test!!!
# @warnings.deprecated("Use spin_currents() instead!")
def calculate_spin_currents(data_grid, direction, fixed_version=False):
    if direction in ["transversal", "y", "trans"] and not fixed_version or direction in ["longitudinal", "x", "long"] and fixed_version:
        slice1 = (slice(0, -1), slice(None))  # equals [:-1, :]
        slice2 = (slice(1, None), slice(None))  # equals [1:, :]
    elif direction in ["longitudinal", "x", "long"] and not fixed_version or direction in ["transversal", "y", "trans"] and fixed_version:
        slice1 = (slice(None), slice(0, -1))  # equals [:, :-1]
        slice2 = (slice(None), slice(1, None))  # equals [:, 1:]
    else:
        raise ValueError(f"Can't compute spin currents in '{direction}'-direction.")

    spin_x_A = select_SL_and_component(data_grid, "A", "x")
    spin_y_A = select_SL_and_component(data_grid, "A", "y")
    spin_x_B = select_SL_and_component(data_grid, "B", "x")
    spin_y_B = select_SL_and_component(data_grid, "B", "y")

    j_inter_1 = - (spin_x_A[slice1] * spin_y_A[slice2] + spin_y_B[slice1] * spin_x_B[slice2])
    j_inter_2 = - (spin_x_A[slice1] * spin_y_A[slice2] - spin_y_B[slice1] * spin_x_B[slice2])
    j_intra_A = - (spin_x_A[slice1] * spin_y_A[slice2] - spin_y_A[slice1] * spin_x_A[slice2])
    j_intra_B = - (spin_x_B[slice1] * spin_y_B[slice2] - spin_y_B[slice1] * spin_x_B[slice2])

    j_other_paper = - spin_x_A[slice1] * spin_y_A[slice2] - spin_y_B[slice1] * spin_x_B[slice2]

    return j_inter_1, j_inter_2, j_intra_A, j_intra_B, j_other_paper


# TODO: Both parts seem to be working   --- if i remember correctly, direction meant direction of Tstep
def calculate_magnetization_neel(data_grid, equi_data_warm=None, direction="x", rel_Tstep_pos=0.49):
    magnetization = dict()
    neel_vector = dict()
    for component in ["x", "y", "z"]:
        magnetization[component] = physics.magnetization(select_SL_and_component(data_grid, "A", component),
                                                         select_SL_and_component(data_grid, "B", component))
        neel_vector[component] = physics.neel_vector(select_SL_and_component(data_grid, "A", component),
                                                     select_SL_and_component(data_grid, "B", component))

    if equi_data_warm is None:      # if no ground state is provided
        return magnetization, neel_vector

    magn_cold = 0
    neel_cold = dict(x=0, y=0, z=1)     # TODO: not sure about that one... For sure does not work with DMI?
    if direction == "x":
        step_pos = helper.get_absolute_T_step_index(rel_Tstep_pos, data_grid.shape[1])
        slice_warm = (slice(None), slice(0, step_pos))
        slice_cold = (slice(None), slice(step_pos, None))
    elif direction == "y":
        step_pos = helper.get_absolute_T_step_index(rel_Tstep_pos, data_grid.shape[0])
        slice_warm = (slice(0, step_pos), slice(None))
        slice_cold = (slice(step_pos, None), slice(None))
    else:
        raise ValueError(f"Can't handle a temperature step in '{direction}'-direction.")

    for component in ["x", "y", "z"]:
        magnetization[component][slice_warm] -= physics.magnetization(
            np.mean(select_SL_and_component(equi_data_warm, "A", component)),
            np.mean(select_SL_and_component(equi_data_warm, "B", component)))  # * 0 - 2.555185061314802e-07
        magnetization[component][slice_cold] -= magn_cold

        neel_vector[component][slice_warm] -= physics.neel_vector(
            np.mean(select_SL_and_component(equi_data_warm, "A", component)),
            np.mean(select_SL_and_component(equi_data_warm, "B", component))
        ) # * 0 - 0.968215418006002
        neel_vector[component][slice_cold] -= neel_cold[component]

    return magnetization, neel_vector



# %% Testing and stuff

"""
TODO:
- Implement function that calculates spin currents, in x and in y direction (longitudinal and transversal)
- Implement function that calculates magnetization and neel vector, also implement possibility (maybe with boolean 
parameter) that decides whether function also subtracts the ground state magnetization for warm and cold region (same 
for neel)
- Implement better plot function (with title, legend etc), also show where temperature step is
- Implement possibility to convolute data or to e.g. average data e.g. in blocks of 2x2 or 4x4 or 8x8
"""

_run =  [2,]

# if __name__ == "__main__" and 0 in _run:
#
#     # %% Testing
#
#     # path1 = "/data/scc/marian.gunsch/AM_tiltedX_Tstep_nernst_T2/spin-configs-99-999/spin-config-99-999-005000.dat"
#     # path2 = "/data/scc/marian.gunsch/AM_tiltedX_ttmstep_7meV_2_id2/spin-configs-99-999/spin-config-99-999-010000.dat"
#     # data1 = read_spin_config_dat(path1)
#     # data2 = read_spin_config_arrjob("/data/scc/marian.gunsch/AM_tiltedX_ttmstep_7meV_2_id",
#     #                                 "/spin-configs-99-999/spin-config-99-999-010000.dat",
#     #                                 10, )
#
#     # path3 = "/data/scc/marian.gunsch/AM-tilted_Tstep_seebeck/spin-configs-99-999/spin-config-99-999-010000.dat"
#     # data3 = read_spin_config_dat(path3)
#     path3_eq = "/data/scc/marian.gunsch/AM_tiltedX_ttmstairs_T2meV/spin-configs-99-999/spin-config-99-999-005000.dat"
#     data3_eq = read_spin_config_dat(path3_eq)
#     #
#     # path4 = "/data/scc/marian.gunsch/AM_tiltedX_Tstep_nernst_T2/spin-configs-99-999/spin-config-99-999-005000.dat"
#     # data4 = read_spin_config_dat(path4)
#     #
#     # path5 = "/data/scc/marian.gunsch/01_AM_tilted_Tstep/spin-configs-99-999/spin-config-99-999-010000.dat"
#     # data5 = read_spin_config_dat(path5)
#     #
#     # path6 = "/data/scc/marian.gunsch/AM_tiltedX_Tstep_nernst_T10/spin-configs-99-999/spin-config-99-999-005000.dat" # high T
#     # path7 = "/data/scc/marian.gunsch/AM_tiltedX_Tstep_nernst_T1/spin-configs-99-999/spin-config-99-999-005000.dat"  # low T
#     #
#     # data6 = read_spin_config_dat(path6)
#     # data7 = read_spin_config_dat(path7)
#     #
#     # path8 = "/data/scc/marian.gunsch/AM-DMI_tilted_Tstep_nernst/spin-configs-99-999/spin-config-99-999-010000.dat"  # DMI
#     # data8 = read_spin_config_dat(path8)
#
#     path9 = "/data/scc/marian.gunsch/02_AM_tilted_Tstep_DMI/spin-configs-99-999/spin-config-99-999-005000.dat"  # DMI more layers
#     data9 = read_spin_config_dat(path9)
#
#     data = data9
#
#     print("Read data")
#
#     # plot_colormap(physics.neel_vector(select_SL_and_component(data1, "A", "z"), select_SL_and_component(data1, "B", "z")), "neel, 1")
#     # plot_colormap(physics.magnetizazion(select_SL_and_component(data1, "A", "z"), select_SL_and_component(data1, "B", "z")), "magn, 1")
#     # plot_colormap(physics.neel_vector(select_SL_and_component(data2, "A", "z"), select_SL_and_component(data2, "B", "z")), "neel, 2")
#     # plot_colormap(physics.magnetizazion(select_SL_and_component(data2, "A", "z"), select_SL_and_component(data2, "B", "z")), "magn, 2")
#
#     # %%
#     rel_Tstep_pos = 0.49
#     show_step = False
#     zoom = False
#
#
#     # magn, neel = calculate_magnetization_neel(data4, data3_eq, "x")
#     magn, neel = calculate_magnetization_neel(data, direction="x")
#     # magn, neel = calculate_magnetization_neel(data1, direction="x")
#     plot_colormap(convolute(average_z_layers(magn["z"]), filter="denoise"), "magnetization z", rel_Tstep_pos, show_step,
#                   zoom)
#     plot_colormap(convolute(average_z_layers(neel["z"]), filter="denoise"), "neel vector z", rel_Tstep_pos, show_step,
#                   zoom)
#
#     #%%
#     # direction = "longitudinal"
#     direction = "transversal"
#
#     # j_inter_1, j_inter_2, j_intra_A, j_intra_B, j_other_paper = calculate_spin_currents(average_z_layers(data), direction)    # This yields incorrect results (or rather all the spin currents are averaged out before being calculated...)
#     j_inter_1, j_inter_2, j_intra_A, j_intra_B, j_other_paper = average_z_layers(*calculate_spin_currents(data, direction))
#
#     # plot_colormap(convolute(j_inter_1), f"j inter +, {direction}", rel_Tstep_pos)
#     # plot_colormap(convolute(j_inter_2), f"j inter -, {direction}", rel_Tstep_pos)
#     # plot_colormap(convolute(j_intra_A), f"j intra A, {direction}", rel_Tstep_pos)
#     # plot_colormap(convolute(j_intra_B), f"j intra B, {direction}", rel_Tstep_pos)
#     # plot_colormap(convolute(j_other_paper), f"j other paper, {direction}", rel_Tstep_pos)
#
#     plot_colormap(j_inter_1, f"j inter +, {direction}", rel_Tstep_pos, show_step, zoom)
#     plot_colormap(j_inter_2, f"j inter -, {direction}", rel_Tstep_pos, show_step, zoom)
#     plot_colormap(j_intra_A, f"j intra A, {direction}", rel_Tstep_pos, show_step, zoom)
#     plot_colormap(j_intra_B, f"j intra B, {direction}", rel_Tstep_pos, show_step, zoom)
#     plot_colormap(j_other_paper, f"j other paper, {direction}", rel_Tstep_pos, show_step, zoom)
#
#
#     # %%
#
#     magnon_count_A = select_SL_and_component(data, "A", "x") ** 2 + select_SL_and_component(data, "A", "y") ** 2
#     magnon_count_B = select_SL_and_component(data, "B", "x") ** 2 + select_SL_and_component(data, "B", "y") ** 2
#
#     plot_colormap(average_z_layers(magnon_count_A), "magnon count A", rel_Tstep_pos, show_step, zoom)
#     plot_colormap(average_z_layers(magnon_count_B), "magnon count B", rel_Tstep_pos, show_step, zoom)
#
#
# if __name__ == "__main__" and 1 in _run:
#
#     # %% Testing not-tilted file
#     non_tilted_path = "/data/scc/marian.gunsch/08_yTstep/T4/spin-configs-99-999/spin-config-99-999-005000.dat"
#
#     data = read_spin_config_dat(non_tilted_path, False)
#
#     # tilted_path = "/data/scc/marian.gunsch/08_tilted_yTstep/T4/spin-configs-99-999/spin-config-99-999-000000.dat"
#     #
#     # data = read_spin_config_dat(tilted_path, True)
#
#
# if __name__ == "__main__" and 2 in _run:
#     np.random.default_rng(462)
#     data = 20 * (np.random.rand(8, 8) - 0.5)
#
#     # print(average_aligned_data(data, "default", False, False))
#     print(average_aligned_data(data, "default", False, True))
#     # print(average_aligned_data(data, "default", True, False))
#     print(average_aligned_data(data, "default", True, True))
