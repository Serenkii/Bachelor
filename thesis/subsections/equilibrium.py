import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.colors as colors

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

def create_figure(figsize=(6.3, 5), max_fig_frac=0.77):
    # ---- replace your fig.add_gridspec(...) with this ----
    fig = plt.figure(figsize=figsize)  # arbitrary figsize

    # your relative units in the gridspec
    left_unit = 1.0  # left narrow column
    quad_unit = 4.0  # each of the two main columns / rows
    bottom_unit = 1.0  # bottom small row

    width_units = left_unit + 2 * quad_unit  # total width units in the gridspec
    height_units = 2 * quad_unit + bottom_unit  # total height units in the gridspec

    fig_w, fig_h = fig.get_size_inches()

    # Start by trying to occupy most of the figure width
    Wg = fig_w * max_fig_frac
    Hg = Wg * (height_units / width_units)  # required height to make central block square

    # If computed Hg is too tall for the figure, instead limit by figure height
    if Hg > fig_h * max_fig_frac:
        Hg = fig_h * max_fig_frac
        Wg = Hg * (width_units / height_units)

    # convert to fractions in figure coordinates for left/right/top/bottom
    left_frac = (fig_w - Wg) / 2.0 / fig_w
    right_frac = left_frac + Wg / fig_w
    bottom_frac = (fig_h - Hg) / 2.0 / fig_h
    top_frac = bottom_frac + Hg / fig_h

    gs = fig.add_gridspec(
        nrows=3, ncols=3,
        width_ratios=[left_unit, quad_unit, quad_unit],
        height_ratios=[quad_unit, quad_unit, bottom_unit],
        left=left_frac, right=right_frac,
        bottom=bottom_frac, top=top_frac,
        hspace=0.05, wspace=0.05
    )
    # ------------------------------------------------------
    return fig, gs



def broken_axes_boundary_plot(bottom_x, bottom_y, left_x, left_y, x_min=0, x_max=256, distance=10,
                              x1_label="", x2_label="", y_label=""):
    shared_kwargs = dict(marker="o", linestyle="--", linewidth=0.7)

    fig, gs = create_figure()

    # Main colorplot: 2x2 axes
    ax00 = fig.add_subplot(gs[0, 1])  # top-left main quadrant (high y, low x)
    ax01 = fig.add_subplot(gs[0, 2])  # top-right main quadrant (high y, high x)
    ax10 = fig.add_subplot(gs[1, 1])  # bottom-left main quadrant (low  y, low  x)
    ax11 = fig.add_subplot(gs[1, 2])  # bottom-right main quadrant (low  y, high x)

    # Left broken 1D (stacked, share y with the corresponding main quadrant row)
    ax_left_top = fig.add_subplot(gs[0, 0], sharey=ax00)
    ax_left_bottom = fig.add_subplot(gs[1, 0], sharey=ax10, sharex=ax_left_top)

    # Bottom broken 1D (side-by-side, share x with the corresponding main quadrant column)
    ax_bottom_left = fig.add_subplot(gs[2, 1], sharex=ax10)
    ax_bottom_right = fig.add_subplot(gs[2, 2], sharex=ax11, sharey=ax_bottom_left)

    def enforce_xlims():
        ax00.set_xlim(x_min, x_min + distance)
        ax00.set_ylim(x_max - distance, x_max)
        ax01.set_xlim(x_max - distance, x_max)
        ax01.set_ylim(x_max - distance, x_max)
        ax10.set_xlim(x_min, x_min + distance)
        ax10.set_ylim(x_min, x_min + distance)
        ax11.set_xlim(x_max - distance, x_max)
        ax11.set_ylim(x_min, x_min + distance)

    def enforce_ylims(min=-0.0099, max=0.0099, ticks=[-0.0075, 0.0, 0.0075]):
        ax_left_top.set_xlim(min, max)
        ax_bottom_left.set_ylim(min, max)
        ax_left_top.set_xticks(ticks)
        ax_bottom_left.set_yticks(ticks)

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
        profile_axs[0].tick_params(bottom=False, labelbottom=False, rotation=-90)
        profile_axs[1].spines.top.set_visible(False)
        profile_axs[1].tick_params(top=False, bottom=True, labelbottom=True, rotation=-90)
        profile_axs[2].spines.right.set_visible(False)
        profile_axs[2].tick_params(right=False, left=False, labelleft=False)
        profile_axs[3].spines.left.set_visible(False)
        profile_axs[3].tick_params(left=False, labelleft=False, right=True, labelright=True)

        enforce_xlims()
        profile_axs[0].set_yticks(profile_axs[3].get_xticks())
        profile_axs[1].set_yticks(profile_axs[2].get_xticks())


    def place_labels(profile_axs, pad=0.055):
        profile_axs[1].set_xlabel(y_label)
        profile_axs[3].yaxis.set_label_position("right")
        profile_axs[3].set_ylabel(y_label)

        x_left = profile_axs[0].get_position().get_points()[0][0]
        y1_left = profile_axs[0].get_position().get_points()[0][1]
        y2_left = profile_axs[1].get_position().get_points()[1][1]
        fig.text(x_left - pad, (y1_left + y2_left) * 0.5, x2_label, rotation=-90, va="center", ha="right")

        pad_bottom = (fig.get_figwidth() * pad) / fig.get_figheight()

        y_bottom = profile_axs[2].get_position().get_points()[0][1]
        x1_bottom = profile_axs[2].get_position().get_points()[1][0]
        x2_bottom = profile_axs[3].get_position().get_points()[0][0]
        fig.text((x1_bottom + x2_bottom) * 0.5, y_bottom - pad_bottom, x1_label, va="top", ha="center")

    def broken_axis_markings():
        size = 7
        util.add_axis_break_marking(ax_left_top, "bottom left", "vertical", size)
        util.add_axis_break_marking(ax_left_top, "bottom right", "vertical", size)
        util.add_axis_break_marking(ax_left_bottom, "top left", "vertical", size)
        util.add_axis_break_marking(ax_left_bottom, "top right", "vertical", size)
        util.add_axis_break_marking(ax_bottom_left, "top right", "horizontal", size)
        util.add_axis_break_marking(ax_bottom_left, "bottom right", "horizontal", size)
        util.add_axis_break_marking(ax_bottom_right, "top left", "horizontal", size)
        util.add_axis_break_marking(ax_bottom_right, "bottom left", "horizontal", size)
        util.add_axis_break_marking(ax00, "top right", "horizontal", size)
        util.add_axis_break_marking(ax00, "bottom left", "vertical", size)
        util.add_axis_break_marking(ax01, "top left", "horizontal", size)
        util.add_axis_break_marking(ax01, "bottom right", "vertical", size)
        util.add_axis_break_marking(ax10, "bottom right", "horizontal", size)
        util.add_axis_break_marking(ax10, "top left", "vertical", size)
        util.add_axis_break_marking(ax11, "bottom left", "horizontal", size)
        util.add_axis_break_marking(ax11, "top right", "vertical", size)


    place_labels((ax_left_top, ax_left_bottom, ax_bottom_left, ax_bottom_right))

    remove_spines_etc((ax_left_top, ax_left_bottom, ax_bottom_left, ax_bottom_right),
                      np.array(((ax00, ax01), (ax10, ax11))))
    enforce_xlims()
    enforce_ylims()

    broken_axis_markings()

    # Left broken 1D plots
    ax_left_top.plot(left_y, left_x, **shared_kwargs)
    ax_left_bottom.plot(left_y, left_x, **shared_kwargs)

    # Bottom broken 1D plots
    ax_bottom_left.plot(bottom_x, bottom_y, **shared_kwargs)
    ax_bottom_right.plot(bottom_x, bottom_y, **shared_kwargs)

    return fig, ax00, ax01, ax10, ax11


def handle_config_data_tilted(config_data1, config_data2):
    magn1 = spinconf_util.average_z_layers(physics.magnetization(
        spinconf_util.select_SL_and_component(config_data1, "A", "z"),
        spinconf_util.select_SL_and_component(config_data1, "B", "z")
    ))
    magn2 = spinconf_util.average_z_layers(physics.magnetization(
        spinconf_util.select_SL_and_component(config_data2, "A", "z"),
        spinconf_util.select_SL_and_component(config_data2, "B", "z")
    ))
    average_magn = np.mean(np.concatenate((magn1, magn2), axis=2), axis=2)
    return average_magn


def place_colorbar(fig, axs, pcolormesh, pad=0.02, width=0.02, label=""):
    # compute bounding box of the 2x2 main grid
    x1 = max(ax.get_position().x1 for ax in axs)
    y0 = min(ax.get_position().y0 for ax in axs)
    y1 = max(ax.get_position().y1 for ax in axs)

    left = x1 + pad
    bottom = y0
    height = y1 - y0

    cax = fig.add_axes([left, bottom, width, height])
    cb = fig.colorbar(pcolormesh, cax=cax)
    cb.set_label(label)
    return cb


def plot_colormap_tilted(fig, axs, x, y, magnetization, y_label=""):
    X, Y = np.meshgrid(x, y, sparse=True, indexing='xy')
    pcms = []
    for ax in axs:
        pcms.append(ax.pcolormesh(X, Y, magnetization, norm=colors.CenteredNorm(), cmap='RdBu_r'))

    vmin = min(p.get_array().min() for p in pcms)
    vmax = max(p.get_array().max() for p in pcms)
    for p in pcms:
        p.set_clim(vmin, vmax)

    place_colorbar(fig, axs, pcms[0], 0.02, 0.02, y_label)
    # fig.subplots_adjust(left=0.1)


def plot_colormap_aligned():
    pass


def boundary_effects():
    profile_suffix = "spin-configs-99-999/mag-profile-99-999.altermagnetA.dat"
    config_suffix = "spin-configs-99-999/spin-config-99-999-005000.dat"

    paths = {
        "100": "/data/scc/marian.gunsch/12/AM_Tstairs_x_T2_openbou/",
        "010": "/data/scc/marian.gunsch/12/AM_Tstairs_y_T2_openbou/",
        "110": "/data/scc/marian.gunsch/04/04_AM_tilted_Tstairs_T2_openbou/",
        "-110": "/data/scc/marian.gunsch/04/04_AM_tilted_yTstairs_T2_openbou/"
    }
    tilted_dict = {
        "100": False,
        "010": False,
        "110": True,
        "-110": True
    }


    # Profile data
    profile_data = dict()
    magnetization_profile = dict()
    real_space = dict()
    for direction in paths:
        profile_data[direction] = mag_util.load_from_path_list([f"{paths[direction]}{profile_suffix}", ],
                                                               which='z', skip_rows=None, return_raw_list=True)

        magnetization_profile[direction] = physics.magnetization(profile_data[direction][0], profile_data[direction][1],
                                                                 True)

        real_space[direction] = np.arange(0.5, magnetization_profile[direction].shape[0], 1.0)
        real_space[direction] = physics.index_to_position(real_space[direction], tilted_dict[direction])

    # Configuration data
    config_data = dict()
    for direction in ["110", "-110"]:
        config_data[direction] = spinconf_util.read_spin_config_dat(f"{paths[direction]}{config_suffix}",
                                                                    is_tilted=True,
                                                                    fixed_version=True)

    # TILTED
    magn_config = handle_config_data_tilted(config_data["110"], config_data["-110"])

    fig_tilted, *axs_tilted = broken_axes_boundary_plot(
        real_space["110"], magnetization_profile["110"],
        real_space["-110"], magnetization_profile["-110"],
        0, real_space["110"][-1] + physics.index_to_position(0.5, True),
        physics.index_to_position(8.2, True),
        r"position $x/a$ in direction \hkl[110]",
        r"position $y/a$ in direction \hkl[-110]",
        r"$\langle S^z \rangle$"
    )

    plot_colormap_tilted(fig_tilted, axs_tilted, real_space["110"], real_space["-110"], magn_config,
                         r"$\langle S^z \rangle$")

    fig_tilted.savefig("out/thesis/boundary_tilted_T2.pdf")

    plt.show()


    # ALIGNED
    axs_aligned = broken_axes_boundary_plot(real_space["100"], magnetization_profile["100"],
                                            real_space["010"], magnetization_profile["010"],
                                            real_space["100"][0], real_space["100"][-1],
                                            physics.index_to_position(10, False))


def main():
    boundary_effects()
