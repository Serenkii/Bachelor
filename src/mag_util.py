import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy as sp

import os

import src.helper as helper
import src.physics as physics

default_slice_dict = {'t': 0, 'x': 1, 'y': 2, 'z': 3, '1': 1, '2': 2, '3': 3}

def time_avg(spin_data):
    return np.average(spin_data, axis=0)


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


def get_components_as_tuple(data, which='xyz', skip_time_steps=0, do_time_avg=False):
    for component in which:
        if do_time_avg:
            yield time_avg(get_component(data, component, skip_time_steps=skip_time_steps))
        else:
            yield get_component(data, component, skip_time_steps=skip_time_steps)


def get_components_as_dict(data, which='xyz', skip_time_steps=0, do_time_avg=False):
    return_dict = dict()
    for component in which:
        if component in return_dict.keys():
            raise ValueError(f"which={which} is invalid because {component} is in it twice!")
        if do_time_avg:
            return_dict[component] = time_avg(get_component(data, component, skip_time_steps=skip_time_steps))
        else:
            return_dict[component] = get_component(data, component, skip_time_steps=skip_time_steps)
    return return_dict


# TODO: Test PROBABLY INDEX ERROR
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
    data_first_A = np.loadtxt(f"{base_path}{start}/{middle_path}A{suffix}")     # TODO: The / could be dangerous!
    data_first_B = np.loadtxt(f"{base_path}{start}/{middle_path}B{suffix}")     # eg if the index is not at the end
    if data_first_A.shape != data_first_B.shape:
        raise ValueError(f"Somehow the dimensions of {base_path}{start}/{middle_path}X{suffix} differ for X=A and X=B.")
    data_arrA = np.empty((array_job_size,) + data_first_A.shape, dtype=data_first_A.dtype)
    data_arrB = np.empty((array_job_size,) + data_first_A.shape, dtype=data_first_A.dtype)
    data_arrA[0] = data_first_A
    data_arrB[0] = data_first_B

    index_list = [start, ]

    for i in range(1, array_job_size):
        job_index = start + i * step
        index_list.append(job_index)
        print(f"{job_index}", end="")
        data_arrA[i] = np.loadtxt(f"{base_path}{job_index}/{middle_path}A{suffix}") # TODO: Just blindly choosing i does not work if start does not start with 0 and step is not 1 etc
        print("A", end="")
        data_arrB[i] = np.loadtxt(f"{base_path}{job_index}/{middle_path}B{suffix}")
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



def get_mean(file_path_A, file_path_B, skip_time_steps=1):
    """
    Returns the spin average of the x, y and z spin component
    :param file_path:
    :param skip_time_steps:
    :return:
    """
    data_A = np.loadtxt(file_path_A)
    data_B = np.loadtxt(file_path_B)
    spins_A = get_components_as_dict(data_A, which="xyz", skip_time_steps=skip_time_steps, do_time_avg=True)
    spins_B = get_components_as_dict(data_B, which="xyz", skip_time_steps=skip_time_steps, do_time_avg=True)

    for component in spins_A:
        spins_A[component] = np.mean(spins_A[component])
        spins_B[component] = np.mean(spins_B[component])

    return spins_A, spins_B



def plot_magnetic_profile(load_paths, skip_rows, save_path, equi_values_warm, equi_values_cold, rel_T_step_positions, plot_kwargs_list):
    """
    Plots and saves the magnetization and neel vector of this magnetic profile. All files are read and plotted in the
    same figure. Equilibrium values can be given and will be subtracted. If none are given, all components are set to
    zero.
    :param load_paths: The magnetic profile paths that will be loaded.
    :param skip_rows: The number of rows that will be skipped when reading the files. (Needed if file is broken)
    :param save_path: The prefix which is used to save the plots.
    :param equi_values_warm: Format: [(dict(x=..., y..., ...), dict(x=..., ...)), ...] for SL A and SL B
    :param equi_values_cold: Format: [(dict(x=..., y..., ...), dict(x=..., ...)), ...] for SL A and SL B
    :param rel_T_step_positions: The relative position of the temperature step. Default is 0.49.
    :param plot_kwargs: The keywords argument for plotting.
    :return: Nothing
    """

    print("Plotting magnetic profile (Magnetization and Neel-Vector)")

    # Checking input
    if not skip_rows:
        skip_rows = 0
    skip_rows = np.array(skip_rows)
    if skip_rows.size == 1:
        skip_rows = np.zeros(len(load_paths), dtype=int) + skip_rows
    if skip_rows.size < len(load_paths):
        raise ValueError("Too few rows in 'skip_rows'.")

    if not rel_T_step_positions:
        rel_T_step_positions = 0.49
    rel_T_step_positions = np.array(rel_T_step_positions)
    if rel_T_step_positions.size == 1:
        rel_T_step_positions = np.zeros(len(load_paths), dtype=float) + 0.49
    if rel_T_step_positions.size < len(load_paths):
        raise ValueError("Too few rows in 'rel_T_step_pos'.")

    if not equi_values_warm and not equi_values_cold:
        equi_values_cold, equi_values_warm = [], []
        zero_dict = dict(x=0, y=0, z=0)
        for _ in load_paths:
            equi_values_cold.append((zero_dict, zero_dict))
            equi_values_warm.append((zero_dict, zero_dict))

    # Read data
    data_A_list, data_B_list = [], []

    for path, skip in zip(load_paths, skip_rows):
        data_A_list.append(np.loadtxt(f"{path}A.dat", skiprows=skip))
        data_B_list.append(np.loadtxt(f"{path}B.dat", skiprows=skip))

    spins_A_list, spins_B_list = [], []

    for data_A, data_B in zip(data_A_list, data_B_list):
        spins_A_list.append(get_components_as_dict(data_A, 'xyz', 1, True))
        spins_B_list.append(get_components_as_dict(data_B, 'xyz', 1, True))

    # Calculating magnetization and neel vectors
    magnetization_list, neel_list = [], []
    for spins_A, spins_B in zip(spins_A_list, spins_B_list):
        magnetization_list.append(dict())
        neel_list.append(dict())
        for component in spins_A.keys():
            magnetization_list[-1][component] = physics.magnetization(spins_A[component], spins_B[component], False)
            neel_list[-1][component] = physics.neel_vector(spins_A[component], spins_B[component], False)


    # Subtracting equilibrium
    magnon_accum_list, delta_neel_list = [], []
    for magnetization, neel_vector, equi_val_cold, equi_val_warm, rel_T_step_pos in zip(magnetization_list, neel_list,
                                                                        equi_values_cold, equi_values_warm, rel_T_step_positions):
        abs_T_step_pos = helper.get_absolute_T_step_index(rel_T_step_pos, magnetization["z"].shape[0])

        magnon_accum_list.append(dict())
        delta_neel_list.append(dict())

        for component in magnetization.keys():
            j = component

            magnetization_warm = physics.magnetization(equi_val_warm[0][component], equi_val_warm[1][component])
            magnetization_cold = physics.magnetization(equi_val_cold[0][component], equi_val_cold[1][component])

            neel_warm = physics.neel_vector(equi_val_warm[0][component], equi_val_warm[1][component])
            neel_cold = physics.neel_vector(equi_val_cold[0][component], equi_val_cold[1][component])

            magnon_accum_list[-1][j] = np.empty_like(magnetization[j])
            magnon_accum_list[-1][j][:abs_T_step_pos] = magnetization[j][:abs_T_step_pos] - magnetization_warm
            magnon_accum_list[-1][j][abs_T_step_pos:] = magnetization[j][abs_T_step_pos:] - magnetization_cold

            delta_neel_list[-1][j] = np.empty_like(neel_vector[j])
            delta_neel_list[-1][j][:abs_T_step_pos] = neel_vector[j][:abs_T_step_pos] - neel_warm
            delta_neel_list[-1][j][abs_T_step_pos:] = neel_vector[j][abs_T_step_pos:] - neel_cold

    # Plotting
    for component in magnon_accum_list[0].keys():
        for quantity_list, title, save_quanti_short in zip((magnon_accum_list, delta_neel_list),
                                                           (f"Magnetization, {component}", f"Neel vector, {component}"),
                                                           ("magn", "neel")):
            print(f"Plotting {component}...")
            fig, ax = plt.subplots()

            ax.set_title(title)
            ax.set_xlabel("Grid position")
            ax.set_ylabel("Magnitude [au]")

            for quantity, plot_kwargs in zip(quantity_list, plot_kwargs_list):
                ax.plot(quantity[component], **plot_kwargs)

            data_list_for_component = []
            for quantity in quantity_list:
                data_list_for_component.append(quantity[component])

            all_data = np.array(data_list_for_component)
            mean = np.mean(all_data)
            median = np.median(all_data)
            mid = (mean + median) / 2
            std = np.std(all_data)
            _max = np.max(all_data)
            _min = np.min(all_data)
            bottom = max(mid - 5 * std, _min - 0.8 * std)
            top = min(mid + 5 * std, _max + 0.8 * std)

            ax.set_ylim(ymin=bottom, ymax=top)

            ax.legend()

            if save_path:
                save_path_ = f"{save_path}_{save_quanti_short}_{component}.pdf"
                print(f"Saving to {save_path_}...")
                plt.savefig(save_path_)

            plt.show()
