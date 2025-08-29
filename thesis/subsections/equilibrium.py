import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import src.utility as util
import src.mag_util as mag_util
import src.bulk_util as bulk_util
import src.utility as util
import src.plot_util as plot_util
import src.physics as physics
import src.spinconf_util as spinconf_util
import src.helper as helper


# %% Introduction of a static magnetic field


# %% boundary effects

def broken_axes_boundary_plot(bottom_x, bottom_y, left_x, left_y, x_min=0, x_max=256, distance=10,
                              x1_label="", x2_label="", y_label=""):
    shared_kwargs = dict(marker="o", linestyle="--", linewidth=0.7)

    fig = plt.figure(figsize=(5, 5))
    gs = fig.add_gridspec(
        nrows=3, ncols=3,
        width_ratios=[1.0, 4.0, 4.0],  # left column narrower, two equal main columns
        height_ratios=[4.0, 4.0, 1.0],  # two main rows and a smaller bottom row
        hspace=0.05, wspace=0.05
    )

    # Main colorplot: 2x2 axes
    ax00 = fig.add_subplot(gs[0, 1])  # top-left main quadrant (high y, low x)
    ax01 = fig.add_subplot(gs[0, 2])  # top-right main quadrant (high y, high x)
    ax10 = fig.add_subplot(gs[1, 1])  # bottom-left main quadrant (low  y, low  x)
    ax11 = fig.add_subplot(gs[1, 2])  # bottom-right main quadrant (low  y, high x)

    def enforce_lims():
        ax00.set_xlim(x_min, x_min + distance)
        ax00.set_ylim(x_max, x_max - distance)
        ax01.set_xlim(x_max, x_max - distance)
        ax01.set_ylim(x_max, x_max - distance)
        ax10.set_xlim(x_min, x_min + distance)
        ax10.set_ylim(x_min, x_min + distance)
        ax11.set_xlim(x_max, x_max - distance)
        ax11.set_ylim(x_min, x_min + distance)

    def remove_spines_etc(profile_axs, colorplot_axs):
        for ax in [colorplot_axs[0, 1], colorplot_axs[1, 1]]:
            ax.tick_params(left=False, labelleft=False)
            ax.spines.left.set_visible(False)
        for ax in [colorplot_axs[0, 0], colorplot_axs[0, 1]]:
            ax.tick_params(bottom=False, labelbottom=False)
            ax.spines.bottom.set_visible(False)
        for ax in [colorplot_axs[0, 0], colorplot_axs[1, 0]]:
            ax.tick_params(left=False, labelleft=False)
            ax.spines.right.set_visible(False)
        for ax in [colorplot_axs[1, 0], colorplot_axs[1, 1]]:
            ax.tick_params(bottom=False, labelbottom=False)
            ax.spines.top.set_visible(False)

        profile_axs[0].spines.bottom.set_visible(False)
        profile_axs[0].tick_params(bottom=False, labelbottom=False)
        profile_axs[1].spines.top.set_visible(False)
        profile_axs[1].tick_params(top=False, bottom=True, labelbottom=True)
        profile_axs[2].spines.right.set_visible(False)
        profile_axs[2].tick_params(right=False, left=False, labelleft=False)
        profile_axs[3].spines.left.set_visible(False)
        profile_axs[3].tick_params(left=False, labelleft=False, right=True, labelright=True)


    # Left broken 1D (stacked, share y with the corresponding main quadrant row)
    ax_left_top = fig.add_subplot(gs[0, 0], sharey=ax00)
    ax_left_bottom = fig.add_subplot(gs[1, 0], sharey=ax10, sharex=ax_left_top)

    # Bottom broken 1D (side-by-side, share x with the corresponding main quadrant column)
    ax_bottom_left = fig.add_subplot(gs[2, 1], sharex=ax10)
    ax_bottom_right = fig.add_subplot(gs[2, 2], sharex=ax11, sharey=ax_bottom_left)

    enforce_lims()
    remove_spines_etc((ax_left_top, ax_left_bottom, ax_bottom_left, ax_bottom_right),
                      np.array(((ax00, ax01), (ax10, ax11))))

    # ax_bottom_right.set_xlabel(x1_label)
    # ax_bottom_right.set_ylabel(y_label)
    # ax_left_bottom.set_xlabel(y_label)
    # ax_left_top.set_ylabel(x2_label)

    # Left broken 1D plots
    ax_left_top.plot(left_y, left_x, **shared_kwargs)
    ax_left_bottom.plot(left_y, left_x, **shared_kwargs)

    # Bottom broken 1D plots
    ax_bottom_left.plot(bottom_x, bottom_y, **shared_kwargs)
    ax_bottom_right.plot(bottom_x, bottom_y, **shared_kwargs)

    return ax00, ax01, ax10, ax11



def boundary_effects():
    profile_suffix = "spin-configs-99-999/mag-profile-99-999.altermagnetA.dat"
    config_suffix = "spin-configs-99-999/spin-config-99-999-005000.dat"

    paths = {
        "100": "/data/scc/marian.gunsch/12/AM_Tstairs_x_T2_openbou/",
        "010": "/data/scc/marian.gunsch/12/AM_Tstairs_y_T2_openbou/",
        "110": "/data/scc/marian.gunsch/04/04_AM_tilted_Tstairs_T2_openbou/",
        "-110": "/data/scc/marian.gunsch/04/04_AM_tilted_yTstairs_T2_openbou/"
    }

    profile_data = dict()
    magnetization_profile = dict()
    real_space = dict()
    for direction in paths:
        profile_data[direction] = mag_util.load_from_path_list(f"{paths[direction]}{profile_suffix}",
                                                               which='z', skip_rows=None)

        magnetization_profile[direction] = physics.magnetization(profile_data[direction][0], profile_data[direction][1],
                                                                 True)

        real_space[direction] = np.arange(magnetization_profile[direction].shape[0])

    # # TODO: REMOVE: dummy data
    # for direction in paths:
    #     real_space[direction] = np.arange(256)
    #     magnetization_profile[direction] = np.cos(real_space[direction])

    real_space["100"] = physics.index_to_position(real_space["100"], False)
    real_space["010"] = physics.index_to_position(real_space["010"], False)
    real_space["110"] = physics.index_to_position(real_space["110"], True)
    real_space["-110"] = physics.index_to_position(real_space["-110"], True)

    axs_tilted = broken_axes_boundary_plot(real_space["110"], magnetization_profile["110"],
                                           real_space["-110"], magnetization_profile["-110"],
                                           real_space["110"][0], real_space["110"][-1],
                                           physics.index_to_position(5, True),
                                           "110", "-110", "S")

    plt.show()

    axs_aligned = broken_axes_boundary_plot(real_space["100"], magnetization_profile["100"],
                                            real_space["010"], magnetization_profile["010"],
                                            real_space["100"][0], real_space["100"][-1],
                                            physics.index_to_position(10, False))


def main():
    boundary_effects()
