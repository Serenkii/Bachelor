import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy as sp



def average_arrayjob_data():
    array_job_size = 10      # TODO: change to 10
    base_path = "/data/scc/marian.gunsch/AM_tiltedX_ttmstep_7meV_2_id"  # index 1 to 10
    middle = "/spin-configs-99-999/mag-profile-99-999.altermagnet"
    suffix = ".dat"
    skip = 1    # number of time steps to skip at the beginning
    data_arrA = np.empty((array_job_size, 10000 - skip, 1537))     # 10 files, 10000 lines & 1537 columns per file
    data_arrB = np.empty((array_job_size, 10000 - skip, 1537))
    print("Loading data...")
    for i in range(array_job_size):
        print(f"{i}", end="")
        data_arrA[i] = np.loadtxt(f"{base_path}{i + 1}{middle}A{suffix}")[skip:, :]
        print("A", end="")
        data_arrB[i] = np.loadtxt(f"{base_path}{i + 1}{middle}B{suffix}")[skip:, :]
        print("B")


    sx_A = np.average(data_arrA[:, :, 1::3], axis=(0, 1))
    sx_B = np.average(data_arrB[:, :, 1::3], axis=(0, 1))
    sy_A = np.average(data_arrA[:, :, 2::3], axis=(0, 1))
    sy_B = np.average(data_arrB[:, :, 2::3], axis=(0, 1))
    sz_A = np.average(data_arrA[:, :, 3::3], axis=(0, 1))
    sz_B = np.average(data_arrB[:, :, 3::3], axis=(0, 1))

    return sx_A, sx_B, sy_A, sy_B, sz_A, sz_B



if __name__ == '__main__':
    sx_A, sx_B, sy_A, sy_B, sz_A, sz_B = average_arrayjob_data()

    neel = 0.5 * (sz_A - sz_B)
    magn = 0.5 * (sz_A + sz_B)

    fig, ax = plt.subplots()
    ax.set_title("Neel")
    ax.plot(neel)

    plt.show()

    fig, ax = plt.subplots()
    ax.set_title("Magn")
    ax.plot(magn)

    plt.show()

    # --> noise seems to be reduced by a about a factor 10

