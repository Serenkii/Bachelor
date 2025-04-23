import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy as sp

array_job_size = 10  # TODO: change to 10
array_job_savefile = "data/nernst/arrj_"

def read_arrayjob_data():
    base_path = "/data/scc/marian.gunsch/AM_tiltedX_ttmstep_7meV_2_id"  # index 1 to 10
    middle = "/spin-configs-99-999/mag-profile-99-999.altermagnet"
    suffix = ".dat"
    data_arrA = np.empty((array_job_size, 10000, 1537))     # 10 files, 10000 lines & 1537 columns per file
    data_arrB = np.empty((array_job_size, 10000, 1537))
    print("Reading original data file...")
    for i in range(array_job_size):
        print(f"{i}", end="")
        data_arrA[i] = np.loadtxt(f"{base_path}{i + 1}{middle}A{suffix}")
        print("A", end="")
        data_arrB[i] = np.loadtxt(f"{base_path}{i + 1}{middle}B{suffix}")
        print("B")

    print(f"Saving data to npy-files... ({array_job_savefile})")
    np.save(f"{array_job_savefile}A.npy", data_arrA)
    np.save(f"{array_job_savefile}B.npy", data_arrB)


def load_arrayjob_npy():
    print("Loading data...")
    A = np.load(f"{array_job_savefile}A.npy")
    print("A", end="")
    B = np.load(f"{array_job_savefile}B.npy")
    print("B")
    return A, B


def array_job_neel_magn(data_arrA, data_arrB, plot=True):
    skip_rows = 1

    spin_z_A_tavg = np.average(data_arrA[:, skip_rows:, 3::3], axis=1)  # time average for each job
    spin_z_B_tavg = np.average(data_arrB[:, skip_rows:, 3::3], axis=1)

    neel_arr = 0.5 * (spin_z_A_tavg - spin_z_B_tavg)
    magn_arr = 0.5 * (spin_z_A_tavg + spin_z_B_tavg)

    neel = np.average(np.abs(neel_arr), axis=0)     # take average over the jobs, use absolute
    magn = np.average(np.abs(magn_arr), axis=0)     # magnitude of vector for each

    if plot:
        fig, ax = plt.subplots()
        ax.set_title("Neel and magnetization")
        ax.set_xlabel("x as spin slot")
        ax.set_ylabel("magnitude")
        ax.plot(neel - 0.9525, label="|Neel| - 0.9525")
        ax.plot(magn, label="|Magn|")
        ax.legend()
        plt.show()

    return neel, magn


def array_job_spin_current(data_arrA, data_arrB, plot=True):
    skip_rows = 1

    spin_x_A = data_arrA[:, skip_rows:, 1::3]
    spin_x_B = data_arrB[:, skip_rows:, 1::3]
    spin_y_A = data_arrA[:, skip_rows:, 2::3]
    spin_y_B = data_arrB[:, skip_rows:, 2::3]

    # intersublattice spin current
    j_inter_1 = - (np.average(spin_x_A[:, :, :-1] * spin_y_A[:, :, 1:], axis=(0, 1))
                   + np.average(spin_y_B[:, :, :-1] * spin_x_B[:, :, 1:], axis=(0, 1)))
    j_inter_2 = - (np.average(spin_x_A[:, :, :-1] * spin_y_A[:, :, 1:], axis=(0, 1))
                   - np.average(spin_y_B[:, :, :-1] * spin_x_B[:, :, 1:], axis=(0, 1)))

    # inrtrasublattice spin current
    j_intra_A = - (np.average(spin_x_A[:, :, :-1] * spin_y_A[:, :, 1:], axis=(0, 1))
                   - np.average(spin_y_A[:, :, :-1] * spin_x_A[:, :, 1:], axis=(0, 1)))
    j_intra_B = - (np.average(spin_x_B[:, :, :-1] * spin_y_B[:, :, 1:], axis=(0, 1))
                   - np.average(spin_y_B[:, :, :-1] * spin_x_B[:, :, 1:], axis=(0, 1)))

    if plot:
        fig, ax = plt.subplots()
        ax.set_title("Spin currents")
        ax.set_xlabel("x as spin slot (or the place in beteen)")
        ax.set_ylabel("magnitude")
        ax.plot(j_inter_1, label="j_inter_1")
        ax.plot(j_inter_2, label="j_inter_2")
        ax.plot(j_intra_A, label="j_intra_A")
        ax.plot(j_intra_B, label="j_intra_B")
        ax.legend()
        plt.show()

    return j_inter_1, j_inter_2, j_intra_A, j_intra_B




if __name__ == '__main__':
    arr_datA, arr_datB = load_arrayjob_npy()
    array_job_neel_magn(arr_datA, arr_datB, plot=True)
    array_job_spin_current(arr_datA, arr_datB, plot=True)

