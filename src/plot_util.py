import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy as sp


def quick_plot_magn_neel(magnetization, neel_vector, info_string="", delta_x=0, save_suffix=None):
    x = np.arange(delta_x, neel_vector - delta_x, 1.0)

    fig, ax = plt.subplots()
    ax.set_xlabel("position (index)")
    ax.set_ylabel("magnitude (au)")
    ax.set_title(f"Neel (~SzA-SzB) ({info_string})")
    ax.plot(x, neel_vector[delta_x:-delta_x])
    if save_suffix:
        print(f"Saving to 'out/Neel_{save_suffix}.png'")
        plt.savefig(f"out/Neel_{save_suffix}.png")
    plt.show()

    fig, ax = plt.subplots()
    ax.set_xlabel("position (index)")
    ax.set_ylabel("magnitude (au)")
    ax.set_title(f"Magn (~SzA+SzB) ({info_string})")
    ax.plot(x, magnetization[delta_x:-delta_x])
    if save_suffix:
        print(f"Saving to 'out/Magn_{save_suffix}.png'")
        plt.savefig(f"out/Magn_{save_suffix}.png")
    plt.show()


def quick_plot_spin_currents(j_inter_1, j_inter_2, j_intra_A, j_intra_B, j_otherpaper, info_string="", save_suffix=None):
    fig, ax = plt.subplots()
    ax.set_title(f"Spin currents ({info_string})")
    ax.set_xlabel("spin index ~ position")
    ax.set_ylabel("magnitude [au]")
    ax.plot(j_inter_1, label="j_inter_+", linewidth=0.6)
    ax.plot(j_inter_2, label="j_inter_-", linewidth=0.6)
    ax.plot(j_intra_A, label="j_intra_A", linewidth=0.6)
    ax.plot(j_intra_B, label="j_intra_B", linewidth=0.6)
    ax.plot(j_otherpaper, label="j_otherpaper", linewidth=0.6)
    ax.legend()
    plt.savefig(f"out/spin_current_{save_suffix}.png")
    plt.show()


