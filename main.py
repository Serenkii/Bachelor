import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy as sp

import src.mag_util as mag_util
import src.bulk_util as bulk_util
import src.utility as util
import src.plot_util as plot_util
import src.physics as physics

Nx = 512
Ny = Nx
Nz = 2
temperature_step_rel_pos = 0.49


def temperature_dependent_neel(save=False, save_path='out/T-dependent-nernst.png', delta_x=0):
    # function does not work here, because not same amount of lines
    # data_dict_A, data_dict_B, npz_path = mag_util.save_arrayjob_as_npz(
    #     "/data/scc/marian.gunsch/AM_tiltedX_Tstep_nernst_T",
    #     "data/nernst/arrj_nernst_Tdependent",
    #     10)

    data_dict_A, data_dict_B = mag_util.load_arrayjob_npyz("data/nernst/arrj_nernst_Tdependent", "npz")

    Sz_A = dict()
    Sz_B = dict()
    neel = dict()
    magn = dict()
    for key in data_dict_A:
        print(key, end="\t")
        Sz_A[key] = util.time_avg(mag_util.get_component(data_dict_A[key], 'z', skip_time_steps=1))
        Sz_B[key] = util.time_avg(mag_util.get_component(data_dict_B[key], 'z', skip_time_steps=1))
        neel[key] = physics.neel_vector(Sz_A[key], Sz_B[key])
        magn[key] = physics.magnetizazion(Sz_A[key], Sz_B[key])
    print()

    x = np.arange(delta_x, neel['1'].size - delta_x, 1.0)

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle("Temperature dependence of nernst effect")
    ax1.set_title("Neel-Vector")
    ax2.set_title("Magnetization")
    ax1.set_ylabel("magnitude (au)")
    ax2.set_ylabel("magnitude (au)")
    # ax1.set_xlabel('Grid position')
    ax2.set_xlabel('Grid position')
    for key in data_dict_A:
        temperature_str = f"T={key}meV"
        if delta_x > 0:
            ax1.plot(x, neel[key][delta_x:-delta_x], label=temperature_str)
            ax2.plot(x, magn[key][delta_x:-delta_x], label=temperature_str)
            continue
        ax1.plot(x, neel[key], label=temperature_str)
        ax2.plot(x, magn[key], label=temperature_str)
    # ax1.legend()
    ax2.legend()
    if save:
        print(f"Saving to '{save_path}'")
        plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    temperature_dependent_neel(save=True, save_path='out/T-dependent-nernst_.png', delta_x=5)

