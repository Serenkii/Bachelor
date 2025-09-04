import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy as sp

import os

import src.helper as helper
import src.physics as physics

default_slice_dict = {'t': 0, 'x': 1, 'y': 2, 'z': 3, '1': 1, '2': 2, '3': 3}

default_force_overwrite = False


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


def get_components_as_tuple(data, which='xyz', skip_time_steps=150, do_time_avg=False):
    for component in which:
        if do_time_avg:
            yield time_avg(get_component(data, component, skip_time_steps=skip_time_steps))
        else:
            yield get_component(data, component, skip_time_steps=skip_time_steps)


def get_components_as_dict(data, which='xyz', skip_time_steps=150, do_time_avg=False):
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
                         force=default_force_overwrite):
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
    data_first_A = np.loadtxt(f"{base_path}{start}/{middle_path}A{suffix}")  # TODO: The / could be dangerous!
    data_first_B = np.loadtxt(f"{base_path}{start}/{middle_path}B{suffix}")  # eg if the index is not at the end
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
        data_arrA[i] = np.loadtxt(
            f"{base_path}{job_index}/{middle_path}A{suffix}")  # TODO: Just blindly choosing i does not work if start does not start with 0 and step is not 1 etc
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
                         force=default_force_overwrite):
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


def get_mean(file_path_A, file_path_B=None, skip_time_steps=150):
    """
    Returns the spin average of the x, y and z spin component
    :param file_path_A: File path of the magnetic profile of SL A. (Or prefix of filename if file_path_B is None)
    :param file_path_B: File path of the magnetic profile of SL B. (If file_path_A is None the prefix of file_path_A is used for both.)
    :param skip_time_steps:
    :return:
    """
    if not file_path_B:
        if file_path_A.endswith("A.dat"):
            file_path_B = f"{file_path_A[:-5]}B.dat"
        else:
            file_path_B = f"{file_path_A}B.dat"
            file_path_A = f"{file_path_A}A.dat"
    data_A = np.loadtxt(file_path_A)
    data_B = np.loadtxt(file_path_B)
    spins_A = get_components_as_dict(data_A, which="xyz", skip_time_steps=skip_time_steps, do_time_avg=True)
    spins_B = get_components_as_dict(data_B, which="xyz", skip_time_steps=skip_time_steps, do_time_avg=True)

    for component in spins_A:
        spins_A[component] = np.mean(spins_A[component])
        spins_B[component] = np.mean(spins_B[component])

    return spins_A, spins_B


def load_from_path_list(load_paths, skip_rows=None, which="xyz", return_raw_list=False):
    """
    Loads the specified components from a list of paths.
    :param load_paths: The magnetic profile paths that will be loaded.
    :param skip_rows: The number of rows that will be skipped when reading the files. (Needed if file is broken)
    :param which: Which components are plotted
    :param return_raw_list: If 'which' only contains one component, setting this to True will return a list of arrays instead of a list of dictionaries containing arrays.
    :return: A list of dictionaries with the keys of 'which'. Each entry in the dictionaries contains an array with the data for the specific component.
    """
    # Checking input
    if not skip_rows:
        skip_rows = 0
    skip_rows = np.array(skip_rows)
    if skip_rows.size == 1:
        skip_rows = np.zeros(len(load_paths), dtype=int) + skip_rows
    if skip_rows.size < len(load_paths):
        raise ValueError("Too few rows in 'skip_rows'.")

    # Read data
    data_A_list, data_B_list = [], []

    for path, skip in zip(load_paths, skip_rows):
        path = path[:-5] if path.endswith(".dat") else path
        data_A_list.append(np.loadtxt(f"{path}A.dat", skiprows=skip))
        data_B_list.append(np.loadtxt(f"{path}B.dat", skiprows=skip))

    spins_A_list, spins_B_list = [], []

    if not return_raw_list or len(which) > 1:
        for data_A, data_B in zip(data_A_list, data_B_list):
            spins_A_list.append(get_components_as_dict(data_A, which, 150, True))
            spins_B_list.append(get_components_as_dict(data_B, which, 150, True))
    else:
        for data_A, data_B in zip(data_A_list, data_B_list):
            spins_A_list.append(time_avg(get_component(data_A, which)))
            spins_B_list.append(time_avg(get_component(data_B, which)))

    return spins_A_list, spins_B_list


def plot_magnetic_profile(spins_A_list, spins_B_list, save_path, equi_values_warm, equi_values_cold,
                          rel_T_step_positions,
                          plot_kwargs_list, title_suffix="", dont_calculate_margins=False):
    """

    :param spins_A_list:
    :param spins_B_list:
    :param save_path: The prefix which is used to save the plots.
    :param equi_values_warm: Format: [(dict(x=..., y..., ...), dict(x=..., ...)), ...] for SL A and SL B
    :param equi_values_cold: Format: [(dict(x=..., y..., ...), dict(x=..., ...)), ...] for SL A and SL B
    :param rel_T_step_positions: The relative position of the temperature step. Default is 0.49.
    :param plot_kwargs_list: The keywords argument for plotting.
    :param title_suffix: Suffix to add to the title of the plot.
    :param dont_calculate_margins: If True, do not try to find sensible margins for the plot.
    :return: magnon_acc_list, delta_neel_list
    """
    print("Plotting magnetic profile (Magnetization and Neel-Vector)")

    # Checking input
    if not rel_T_step_positions:
        rel_T_step_positions = 0.49
    rel_T_step_positions = np.array(rel_T_step_positions)
    if rel_T_step_positions.size == 1:
        rel_T_step_positions = np.zeros(len(spins_A_list), dtype=float) + rel_T_step_positions
    if rel_T_step_positions.size < len(spins_A_list):
        raise ValueError("Too few rows in 'rel_T_step_pos'.")

    if not equi_values_warm:
        equi_values_warm = []
        zero_dict = dict(x=0, y=0, z=0)
        for _ in spins_A_list:
            equi_values_warm.append((dict(x=0, y=0, z=1), dict(x=0, y=0, z=-1)))

    if not equi_values_cold:
        equi_values_cold = []
        zero_dict = dict(x=0, y=0, z=0)
        for _ in spins_A_list:
            equi_values_cold.append((dict(x=0, y=0, z=1), dict(x=0, y=0, z=-1)))

    # Calculating magnetization and neel vectors
    magnetization_list, neel_list = [], []
    for spins_A, spins_B in zip(spins_A_list, spins_B_list):
        magnetization_list.append(dict())
        neel_list.append(dict())
        for component in spins_A.keys():
            magnetization_list[-1][component] = physics.magnetization(spins_A[component], spins_B[component], False)
            neel_list[-1][component] = physics.neel_vector(spins_A[component], spins_B[component], False)

    def calculate_margins(data_list_for_component):
        max_len = max(len(sub) for sub in data_list_for_component)
        all_data = np.full((len(data_list_for_component), max_len), np.nan)
        for i, sub in enumerate(data_list_for_component):
            all_data[i, :len(sub)] = sub

        mean = np.nanmean(all_data)
        median = np.nanmedian(all_data)
        mid = (mean + median) / 2
        std = np.nanstd(all_data)
        _max = np.nanmax(all_data)
        _min = np.nanmin(all_data)
        bottom = max(mid - 5 * std, _min - 0.8 * std)
        top = min(mid + 5 * std, _max + 0.8 * std)
        return (bottom, top)

    # Subtracting equilibrium
    magnon_accum_list, delta_neel_list = [], []

    for magnetization, neel_vector, equi_val_cold, equi_val_warm, rel_T_step_pos in zip(magnetization_list, neel_list,
                                                                                        equi_values_cold,
                                                                                        equi_values_warm,
                                                                                        rel_T_step_positions):
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
                                                           (f"Magnetization, {component} {title_suffix}",
                                                            f"Neel vector, {component} {title_suffix}"),
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

            if not dont_calculate_margins:
                bottom, top = calculate_margins(data_list_for_component)
                ax.set_ylim(ymin=bottom, ymax=top)

            ax.legend()

            if save_path:
                save_path_ = f"{save_path}_{save_quanti_short}_{component}.pdf"
                print(f"Saving to {save_path_}...")
                plt.savefig(save_path_)

            plt.show()

    return magnon_accum_list, delta_neel_list


def plot_magnetic_profile_from_paths(load_paths, skip_rows, save_path, equi_values_warm, equi_values_cold,
                                     rel_T_step_positions, plot_kwargs_list, title_suffix="",
                                     dont_calculate_margins=False, which="xyz"):
    """
    Plots and saves the magnetization and neel vector of the magnetic profiles. All files are read and plotted in the
    same figure. Equilibrium values can be given and will be subtracted. If none are given, all equilibrium components
    are set to zero, therefore subtracting does not change the outcome.
    :param load_paths: The magnetic profile paths that will be loaded.
    :param skip_rows: The number of rows that will be skipped when reading the files. (Needed if file is broken)
    :param save_path: The prefix which is used to save the plots.
    :param equi_values_warm: Format: [(dict(x=..., y..., ...), dict(x=..., ...)), ...] for SL A and SL B
    :param equi_values_cold: Format: [(dict(x=..., y..., ...), dict(x=..., ...)), ...] for SL A and SL B
    :param rel_T_step_positions: The relative position of the temperature step. Default is 0.49.
    :param plot_kwargs_list: The keywords argument for plotting.
    :param title_suffix: Suffix to add to the title of the plot.
    :param dont_calculate_margins: If True, do not try to find sensible margins for the plot.
    :param which: Which components are plotted. Default is 'xyz' for 'x', 'y' and 'z'.
    :return: spins_A_list, spins_B_list, magnon_acc_list, delta_neel_list
    """

    spins_A_list, spins_B_list = load_from_path_list(load_paths, skip_rows, which)

    magnon_acc_list, delta_neel_list = plot_magnetic_profile(
        spins_A_list, spins_B_list, save_path, equi_values_warm, equi_values_cold,
        rel_T_step_positions, plot_kwargs_list, title_suffix, dont_calculate_margins)

    return spins_A_list, spins_B_list, magnon_acc_list, delta_neel_list


def save_mag_files(file_path_A, save_path_prefix, file_path_B=None, saving_after_index=0,
                   force=default_force_overwrite):
    """
    Loads the text files in the specified (paths) and saves them in the save_path. Only saves from line/timestep
    saving_after_index. If force is False (default) no files will be loaded or saved, if the save files exist already.
    :param file_path_A: The path of the magnetic profile of SL A.
    :param save_path_prefix: The prefix of the save_path. The output path then is something like save_path + '_{SL}.npy'
    :param file_path_B: The path of the magnetic profile of SL B. If not specified, it is attempted to guess the path from file_path_A.
    :param saving_after_index: Line index / time step after which the magnetic profile will be saved. If you want to save the last 100 lines, specify -100 e.g.
    :param force: If True, existing save_files will be overwritten.
    :return: The numpy save paths.
    """
    file_path_B = file_path_B or f"{file_path_A[:-5]}B.dat"

    save_path_A = f"{save_path_prefix}_A.npy" if not save_path_prefix.endswith("_A.npy") else save_path_prefix
    save_path_B = f"{save_path_prefix[:-6]}_B.npy"

    if os.path.isfile(save_path_A) or os.path.isfile(save_path_B) and not force:
        print(f"File {save_path_A} or {save_path_B} already exists, no file is overwritten, therefore skipping.")
        return save_path_A, save_path_B

    print(f"Loading data from {file_path_A} and {file_path_B}...", end=" ")
    dataA = np.loadtxt(file_path_A)
    print("A", end=".")
    dataB = np.loadtxt(file_path_B)
    print("B.")

    print(f"Saving data to {save_path_A} and {save_path_B}...")

    np.save(save_path_A, dataA[saving_after_index:])
    np.save(save_path_B, dataB[saving_after_index:])

    return save_path_A, save_path_B


def load_mag_npy_files(path_A, path_B=None):
    """

    :param path_A:
    :param path_B:
    :return:
    """
    path_A = path_A if path_A.endswith("_A.npy") else f"{path_A}_A.npy"
    path_B = path_B or f"{path_A[:-6]}_B.npy"
    print(f"Loading data from {path_A} and {path_B}...")
    return np.load(path_A), np.load(path_B)


def infer_path_B(path, also_return_path_A=False):
    if path.endswith(".dat"):
        path = path[:-5]

    path_B = f"{path}B.dat"

    if also_return_path_A:
        return f"{path}A.dat", path_B

    return path_B


def get_dummy_data(size=512, time_steps=100000, sublatticeA=True, components="z", as_dict=False, do_time_avg=False,
                   seed=None):
    print("WARNING!\n"
          "using dummy data...\n\n")

    sign = +1 if sublatticeA else -1

    rng = np.random.default_rng(seed)

    def generate_data(component):
        if component == "x" or component == "y":
            return rng.uniform(-0.05, 0.05, (time_steps, size))
        if component == "z":
            return sign * rng.uniform(0.8, 1.0, (time_steps, size))

    func = time_avg if do_time_avg else lambda arr: arr

    if as_dict:
        return_dict = dict()
        for component in components:
            return_dict[component] = func(generate_data(component))
        return return_dict

    if len(components) == 1:
        return func(generate_data(components))

    for component in components:
        yield func(generate_data(component))


# Untested
def infer_data_path(path, also_return_path_B=False):
    folder_list = path.split("/")
    if folder_list[-1] == "":       # path ends with '/'
        folder_list.pop()
    else:
        path += "/"

    return_list = []

    if folder_list[-1].endswith(".dat"):
        return_list.append(f"{path[:-5]}A.dat")
        if also_return_path_B:
            return_list.append(f"{path[:-5]}B.dat")

    elif folder_list[-1] in ["mag-profile-99-999.altermagnetA", "mag-profile-99-999.altermagnetB"]:
        return_list.append(f"{path[:-1]}A.dat")
        if also_return_path_B:
            return_list.append(f"{path[:-1]}B.dat")

    elif folder_list[-1] == "spin-configs-99-999":
        return_list.append(f"{path}mag-profile-99-999.altermagnetA.dat")
        if also_return_path_B:
            return_list.append(f"{path}mag-profile-99-999.altermagnetB.dat")

    else:
        return_list.append(f"{path}spin-configs-99-999/mag-profile-99-999.altermagnetA.dat")
        if also_return_path_B:
            return_list.append(f"{path}spin-configs-99-999/mag-profile-99-999.altermagnetB.dat")

    for entry in return_list:
        if not os.path.exists(entry):
            raise OSError(f"Unable to infer data file from '{path}'. Unsuccessful attempt yielded '{entry}'.")

    return tuple(return_list)


# Untested
def npy_files(dat_path: str, npy_path=None, slice_index=-100000, force=default_force_overwrite, return_data=True,
              **load_kwargs):
    if "max_rows" not in load_kwargs.keys():
        load_kwargs["max_rows"] = 1_000_000

    base_folder = "data/profiles/"

    data_path_A, data_path_B = infer_data_path(dat_path, True)

    if not npy_path:
        folder_list = data_path_A.split("/")
        index0 = folder_list.index("marian.gunsch")
        save_name = f"{folder_list[index0 + 1].zfill(2)}_{folder_list[index0 + 2]}"
        npy_path = f"{base_folder}{save_name}"
        print(f"Chose npy base path '{npy_path}' based on the given data path.")
    elif npy_path.endswith(".npy"):
        npy_path = npy_path[:-4]

    npy_path_A = f"{npy_path}.A.npy"
    npy_path_B = f"{npy_path}.B.npy"

    data_dict = dict(A=None, B=None)

    for npy_path, dat_path, SL in zip((npy_path_A, npy_path_B), (data_path_A, data_path_B), ("A", "B")):
        print(f"Handling sublattice {SL}...")
        if os.path.isfile(npy_path) and not force:
            print(f"File {npy_path} already exists. Nothing will be overwritten.")
        else:
            print(f"Reading data from {dat_path}...")
            data = np.loadtxt(dat_path, **load_kwargs)[slice_index:]
            print(f"Saving data to {npy_path}...")
            np.save(npy_path, data)
            if return_data:
                data_dict[SL] = data

    if not return_data:
        return

    for SL, npy_path in zip(data_dict.keys(), (npy_path_A, npy_path_B)):
        if data_dict[SL] is None:
            print(f"{SL}: Loading data from {npy_path}...")
            data_dict[SL] = np.load(npy_path)

    print("\n")

    return data_dict["A"], data_dict["B"]


# Untested
def npy_files_from_dict(path_dict, slice_index=-100000, force=default_force_overwrite, which=None, **load_kwargs):
    """
    Returns a dictionary containing data from specific paths. Used .npy files to read/save.
    :param path_dict:
    :param slice_index: Starting index for slicing data. (Reduce data size)
    :param force: If True, forcefully create new npy-files.
    :param which: If None, then the 'raw' data is returned. If a string like 'z', the data of this component is returned.
    :return: Two dictionaries, one for each sublattice: A, B
    """
    data_A = dict()
    data_B = dict()
    for key in path_dict:
        datA, datB = npy_files(path_dict[key], slice_index=slice_index, force=force, **load_kwargs)
        if which is not None:
            data_A[key] = get_component(datA, which, 0)
            data_B[key] = get_component(datB, which, 0)
        else:
            data_A[key] = datA
            data_B[key] = datB

    return data_A, data_B
