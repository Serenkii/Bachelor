import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy as sp

import src.physics as physics
import src.helper as helper

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

    shape = (lengths['x'], lengths['y'], lengths['z'], number_sublattices, 3)
    value_grid = np.zeros(shape) + 1000  # TODO: Change to np empty and remove addition
    value_grid[j, i, k, sl, 0] = data[:, 4]  # 4=x, 5=y, 6=z
    value_grid[j, i, k, sl, 1] = data[:, 5]  # (components of spin)
    value_grid[j, i, k, sl, 2] = data[:, 6]  # j i k instead of i j k because indecis are weird...

    value_grid_zavg = np.average(value_grid, axis=2)  # average over k (z layers)

    return value_grid_zavg


def save_spin_config_as_npy(dat_path, save_path, is_tilted=True):
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


def plot_colormap(data_grid, title=""):
    X, Y = np.meshgrid(np.arange(0, data_grid.shape[0], 1, dtype=int),
                       np.arange(0, data_grid.shape[1], 1, dtype=int),
                       sparse=True, indexing='xy')
    fig, ax = plt.subplots()
    ax.set_title(title)
    im = ax.pcolormesh(X, Y, data_grid)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    plt.show()


# TODO: Test!!!
def calculate_spin_currents(data_grid, direction):
    if direction in ["longitudinal", "x", "long"]:
        slice1 = (slice(0, -1), slice(None))  # equals [:-1, :]
        slice2 = (slice(1, None), slice(None))  # equals [1:, :]
    elif direction in ["transversal", "y", "trans"]:
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
    j_intra_B = - (spin_x_A[slice1] * spin_y_A[slice2] - spin_y_A[slice1] * spin_x_A[slice2])

    j_other_paper = - spin_x_A[slice1] * spin_y_A[slice2] - spin_y_B[slice1] * spin_x_B[slice2]

    return j_inter_1, j_inter_2, j_intra_A, j_intra_B, j_other_paper


# TODO: Test!!!
def calculate_magnetization_neel(data_grid, equi_data_warm=None, direction="x", rel_Tstep_pos=0.49):
    magnetization = dict()
    neel_vector = dict()
    for component in ["x", "y", "z"]:
        magnetization[component] = physics.magnetizazion(select_SL_and_component(data_grid, "A", component),
                                                         select_SL_and_component(data_grid, "B", component))
        neel_vector[component] = physics.neel_vector(select_SL_and_component(data_grid, "A", component),
                                                     select_SL_and_component(data_grid, "B", component))

    if not equi_data_warm:      # if no ground state is provided
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
        magnetization[component][slice_warm] -= physics.magnetizazion(
            np.mean(select_SL_and_component(equi_data_warm, "A", component)),
            np.mean(select_SL_and_component(equi_data_warm, "B", component))
        )
        magnetization[component][slice_cold] -= magn_cold

        neel_vector[component][slice_warm] -= physics.neel_vector(
            np.mean(select_SL_and_component(equi_data_warm, "A", component)),
            np.mean(select_SL_and_component(equi_data_warm, "B", component))
        )
        neel_vector[component][slice_cold] -= neel_cold[component]

    return magnetization, neel_vector



"""
TODO:
- Implement function that calculates spin currents, in x and in y direction (longitudinal and transversal)
- Implement function that calculates magnetization and neel vector, also implement possibility (maybe with boolean 
parameter) that decides whether function also subtracts the ground state magnetization for warm and cold region (same 
for neel)
- Implement better plot function (with title, legend etc), also show where temperature step is
- Implement possibility to convolute data or to e.g. average data e.g. in blocks of 2x2 or 4x4 or 8x8
"""

# %% Testing

path1 = "/data/scc/marian.gunsch/AM_tiltedX_Tstep_nernst_T2/spin-configs-99-999/spin-config-99-999-005000.dat"
path2 = "/data/scc/marian.gunsch/AM_tiltedX_ttmstep_7meV_2_id2/spin-configs-99-999/spin-config-99-999-010000.dat"
data1 = read_spin_config_dat(path1)
data2 = read_spin_config_arrjob("/data/scc/marian.gunsch/AM_tiltedX_ttmstep_7meV_2_id",
                                "/spin-configs-99-999/spin-config-99-999-010000.dat",
                                10, )

plot_colormap(physics.neel_vector(select_SL_and_component(data1, "A", "z"), select_SL_and_component(data1, "B", "z")))
plot_colormap(physics.magnetizazion(select_SL_and_component(data1, "A", "z"), select_SL_and_component(data1, "B", "z")))
plot_colormap(physics.neel_vector(select_SL_and_component(data2, "A", "z"), select_SL_and_component(data2, "B", "z")))
plot_colormap(physics.magnetizazion(select_SL_and_component(data2, "A", "z"), select_SL_and_component(data2, "B", "z")))
