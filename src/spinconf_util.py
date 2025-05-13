import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy as sp


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


def read_spin_config_dat(path, is_tilted=True):
    if not is_tilted:
        raise NotImplementedError("Non-tilted is untested!")

    dimensions, lengths = get_dimensions(path)
    divisors = dict(x=int(dimensions['x']/lengths['x']), y=int(dimensions['y']/lengths['y']), z=int(dimensions['z']/lengths['z']))

    data = np.loadtxt(path)

    data_mins = np.min(data[:, :3], axis=0)
    data[:, :3] -= data_mins

    SL_A = np.where(np.expand_dims(data[:, 3] == 1.0, 1), data, np.nan)
    SL_A = SL_A[~np.isnan(SL_A).any(axis=1)]        # remove all rows that are not sublattice A
    SL_B = np.where(np.expand_dims(data[:, 3] == 2.0, 1), data, np.nan)
    SL_B = SL_B[~np.isnan(SL_B).any(axis=1)]

    # SL_A = np.where(np.expand_dims(data[:, 3] == 1.0, 1), data, np.nan)  # SL A
    # SL_B = np.where(np.expand_dims(data[:, 3] == 2.0, 1), data, np.nan)  # SL A
    # SL_A_maxs = np.nanmax(SL_A[:, :3], axis=0)
    # SL_A_mins = np.nanmin(SL_A[:, :3], axis=0)
    # SL_B_maxs = np.nanmax(SL_B[:, :3], axis=0)
    # SL_B_mins = np.nanmin(SL_B[:, :3], axis=0)
    #
    # SL_A[:, :3] -= SL_A_mins
    # SL_B[:, :3] -= SL_B_mins


    X, Y = np.meshgrid(np.arange(0, lengths['x'], 1, dtype=int),
                       np.arange(0, lengths['y'], 1, dtype=int),
                       sparse=True)

    i, j, k = SL_A[:, 0].astype(int) // divisors['x'], SL_A[:, 1].astype(int) // divisors['y'], SL_A[:, 2].astype(int) // divisors['z']

    shape = (lengths['x'], lengths['y'], lengths['z'])
    value_array = np.zeros(shape)
    value_array[i, j, k] = SL_A[:, 4]       # x values

    fig, ax = plt.subplots()
    ax.pcolormesh(X, Y, value_array[:, :, 0])

    plt.show()

    return value_array, SL_A

def save_spin_config_as_npy(path):
    pass



# %% Testing

path = "/data/scc/marian.gunsch/AM_tiltedX_ttmstep_7meV_2_id2/spin-configs-99-999/spin-config-99-999-010000.dat"

temp1, temp2 = read_spin_config_dat(path)


