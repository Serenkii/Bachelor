import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy as sp


number_spins = 512
rel_pos_Tstep = 0.49
abs_pos_Tstep = int(number_spins * rel_pos_Tstep)

def read_seebeck(base_path=None, save_path=None):
    if not base_path:
        base_path = "/data/scc/marian.gunsch/AM-DMI_tilted_Tstep_seebeck/spin-configs-99-999/mag-profile-99-999.altermagnet"
    if not save_path:
        save_path = "data/DMI/seebeck/single_DMIseeb.npz"
    suffix = ".dat"
    print("Reading original data file...")
    dataA = np.loadtxt(f"{base_path}A{suffix}")
    print("A", end="")
    dataB = np.loadtxt(f"{base_path}B{suffix}")
    print("B")
    print(f"Saving data to {save_path}")
    np.savez(f"{save_path}", A=dataA, B=dataB)


def load_seebeck(path=None):
    if not path:
        path = "data/DMI/seebeck/single_DMIseeb.npz"
    print(f"Loading data from '{path}'...")
    return np.load(path)


def read_equilibrium_DMI(path=None, save_path=None, number_time_steps=3000):
    if not path:
        path = "/data/scc/marian.gunsch/AM_tiltedX_ttmstairs_DMI_ferri-2/AM_Teq-99-999.dat"
    if not save_path:
        save_path = "data/DMI/seebeck/equi_T2meV_DMI5.npy"
    print(f"Reading equilibrium data from '{path}'...")
    data = np.loadtxt(path)
    print(f"Saving equilibrium data to '{save_path}'...")
    np.save(f"{save_path}", data[-number_time_steps:, :])


def load_equilibrium_DMI(path=None):
    if not path:
        path = "data/DMI/seebeck/equi_T2meV_DMI5.npy"
    print(f"Loading data from '{path}'...")
    return np.load(path)


def magn_neel_seebeck(Tstep_data_npz, equi_data, plot=True):
    # equilibriums
    Sz_A_2eq = np.average(equi_data[:, 5], axis=0)  # average time
    Sz_B_2eq = np.average(equi_data[:, 8], axis=0)

    neel_2eq = 0.5 * (Sz_A_2eq - Sz_B_2eq)
    magn_2eq = 0.5 * (Sz_A_2eq + Sz_B_2eq)
    neel_0eq = 1
    magn_0eq = 0

    neel_eq = np.empty_like(Tstep_data_npz["A"][0, 1::3])
    neel_eq[0:abs_pos_Tstep] = neel_2eq
    neel_eq[abs_pos_Tstep:] = neel_0eq

    magn_eq = np.empty_like(Tstep_data_npz["A"][0, 1::3])
    magn_eq[0:abs_pos_Tstep] = magn_2eq
    magn_eq[abs_pos_Tstep:] = magn_0eq

    Sz_A = np.average(Tstep_data_npz["A"][1:, 3::3], axis=0)    # average time
    Sz_B = np.average(Tstep_data_npz["B"][1:, 3::3], axis=0)

    # neel
    neel = 0.5 * (Sz_A - Sz_B)
    magn = 0.5 * (Sz_A + Sz_B)

    # subtracting
    delta_neel = neel - neel_eq
    delta_magn = magn - magn_eq

    if plot:
        fig, ax = plt.subplots()
        ax.set_title("Spin Seebeck with DMI")
        ax.plot(delta_neel, label = "Delta neel")
        ax.plot(delta_magn, label = "Delta magn")
        ax.legend()
        plt.show()

    return delta_neel, delta_magn

if __name__ == '__main__':
    npzfile = load_seebeck()
    equilibrium_DMI = load_equilibrium_DMI()

    d_magn_seeb, d_neel_seeb = magn_neel_seebeck(npzfile, equilibrium_DMI)


