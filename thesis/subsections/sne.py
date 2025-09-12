import warnings

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.colors as colors
from matplotlib.collections import PolyCollection
from sympy.printing.pretty.pretty_symbology import line_width

import src.utility as util
import src.mag_util as mag_util
import src.bulk_util as bulk_util
import src.utility as util
import src.plot_util as plot_util
import src.physics as physics
import src.spinconf_util as spinconf_util
import src.helper as helper
import thesis.mpl_configuration as mpl_config

# %% Spin accumulation / net magnetization

save_base_path = "out/thesis/sne/"


# See here e.g. for T-dependence: /data/scc/marian.gunsch/00/AM_tiltedX_Tstep_nernst/

def spin_accu_config_plot(magnetization, x_space, y_space, magn_profile_x, magn_profile_y,
                          x_min=116.6, x_max=141.4, distance=8.4, vmin=-0.0099, vmax=0.0099,
                          magn_label=r"$\langle S^z \rangle$", x_space_label="", y_space_label="",
                          save_suffix=""):
    profile_kwargs = dict(marker="o", linestyle="--", linewidth=0.7, markersize=3.0)

    X, Y = np.meshgrid(x_space,
                       y_space,
                       sparse=True, indexing="xy")

    def create_figure(profile_frac=0.3, margin_frac=0.1,
                      prefer='height', tol=1e-9, maxiter=80, debug=False):
        """
        Create fig + GridSpec such that the central data-block (middle column, top row)
        has physical aspect = x_range / y_range (so data pixels are square).

        See: https://chatgpt.com/share/68c2c9fa-1bfc-8010-a8b8-3f076eb2a7dd

        Parameters
        ----------
        profile_frac : float
            fraction used for the side/bottom profile rows/cols (same as you used)
        margin_frac : float
            fraction used for the right margin/colorbar column
        prefer : 'height' or 'width'
            whether to solve for the data-row height ratio (recommended) or the middle width ratio
        tol : float
            tolerance on achieved aspect ratio (relative)
        maxiter : int
            max iterations for the bisection search
        debug : bool
            print diagnostic info

        Returns
        -------
        fig, gs
        """

        fig = plt.figure(figsize=(mpl_config.get_width(), 0.85 * mpl_config.get_height()))
        fw = fig.get_figwidth()
        fh = fig.get_figheight()

        # margins exactly as you specified
        left_space = 0.19
        right_space = 0.19
        vertical_space = (left_space + right_space) / fw * fh
        top_space = 0.1 * vertical_space
        bottom_space = 0.9 * vertical_space

        # data ranges
        x_range = x_max - x_min
        y_range = distance
        target_aspect = (x_range / y_range)

        # spacing exactly as you specified
        space_betw_axes_vert = 0.05
        space_betw_axes_hori = space_betw_axes_vert * (fw / fh)
        wspace = space_betw_axes_hori
        hspace = space_betw_axes_vert

        # available physical space for the whole GridSpec block (after margins)
        A = fw * (1.0 - left_space - right_space)  # total physical width for gridspec
        B = fh * (1.0 - top_space - bottom_space)  # total physical height for gridspec

        # fixed user-specified parts
        wr0 = profile_frac
        wr2 = margin_frac

        # initial middle width guess (keeps intuitive proportionality)
        wr1_init = target_aspect if target_aspect > 0 else 1.0

        # helper to compute achieved aspect given (wr0, wr1, wr2) and (hr0, hr1, hr2)
        def achieved_aspect(width_ratios, height_ratios):
            denomW = sum(width_ratios) + 2.0 * wspace  # 3 cols => 2 gaps
            denomH = sum(height_ratios) + 2.0 * hspace  # 3 rows => 2 gaps
            W_data = (width_ratios[1] / denomW) * A
            H_data = (height_ratios[0] / denomH) * B
            return W_data / H_data

        # Attempt 1: solve for data-row height h (both data rows equal = h)
        def solve_for_height(wr1):
            # bracket search for h producing correct aspect
            # We expect h to be positive; use bisection over a wide range
            h_low, h_high = 1e-6, 1e6

            # quick checks for sign change
            hr_low = [h_low, h_low, profile_frac]
            hr_high = [h_high, h_high, profile_frac]
            ar_low = achieved_aspect([wr0, wr1, wr2], hr_low) - target_aspect
            ar_high = achieved_aspect([wr0, wr1, wr2], hr_high) - target_aspect

            if ar_low == 0:
                return h_low
            if ar_high == 0:
                return h_high

            # if no sign change, the equation might not be solvable for this wr1
            if ar_low * ar_high > 0:
                return None

            # bisection
            for i in range(maxiter):
                h_mid = 0.5 * (h_low + h_high)
                ar_mid = achieved_aspect([wr0, wr1, wr2], [h_mid, h_mid, profile_frac]) - target_aspect
                if abs(ar_mid) <= tol * max(1.0, target_aspect):
                    return h_mid
                if ar_low * ar_mid <= 0:
                    h_high = h_mid
                    ar_high = ar_mid
                else:
                    h_low = h_mid
                    ar_low = ar_mid
            # didn't converge
            return None

        # Attempt 2: solve for middle width wr1 (keep both data rows height = 1)
        def solve_for_width(h_try=1.0):
            wr_low, wr_high = 1e-6, 1e6
            hr = [h_try, h_try, profile_frac]
            ar_low = achieved_aspect([wr0, wr_low, wr2], hr) - target_aspect
            ar_high = achieved_aspect([wr0, wr_high, wr2], hr) - target_aspect
            if ar_low == 0:
                return wr_low
            if ar_high == 0:
                return wr_high
            if ar_low * ar_high > 0:
                return None
            for i in range(maxiter):
                wr_mid = 0.5 * (wr_low + wr_high)
                ar_mid = achieved_aspect([wr0, wr_mid, wr2], hr) - target_aspect
                if abs(ar_mid) <= tol * max(1.0, target_aspect):
                    return wr_mid
                if ar_low * ar_mid <= 0:
                    wr_high = wr_mid
                    ar_high = ar_mid
                else:
                    wr_low = wr_mid
                    ar_low = ar_mid
            return None

        width_ratios = None
        height_ratios = None
        solved = False

        # Try preferred method first
        if prefer == 'height':
            wr1 = wr1_init
            h_sol = solve_for_height(wr1)
            if h_sol is not None:
                height_ratios = [h_sol, h_sol, profile_frac]
                width_ratios = [wr0, wr1, wr2]
                solved = True
            else:
                # fallback: solve for width
                wr1_sol = solve_for_width(h_try=1.0)
                if wr1_sol is not None:
                    width_ratios = [wr0, wr1_sol, wr2]
                    height_ratios = [1.0, 1.0, profile_frac]
                    solved = True
        else:  # prefer width
            wr1_sol = solve_for_width(h_try=1.0)
            if wr1_sol is not None:
                width_ratios = [wr0, wr1_sol, wr2]
                height_ratios = [1.0, 1.0, profile_frac]
                solved = True
            else:
                # fallback: solve for height
                wr1 = wr1_init
                h_sol = solve_for_height(wr1)
                if h_sol is not None:
                    height_ratios = [h_sol, h_sol, profile_frac]
                    width_ratios = [wr0, wr1, wr2]
                    solved = True

        # Last-resort fallback (keep simple sensible defaults)
        if not solved:
            if debug:
                print("[create_figure] Warning: exact solution not found; falling back to defaults.")
            width_ratios = [wr0, wr1_init, wr2]
            height_ratios = [1.0, 1.0, profile_frac]

        # Build the GridSpec
        gs = fig.add_gridspec(
            nrows=3,
            ncols=3,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            wspace=wspace,
            hspace=hspace,
            left=left_space,
            bottom=bottom_space,
            right=1.0 - right_space,
            top=1.0 - top_space
        )

        # diagnostics
        ach = achieved_aspect(width_ratios, height_ratios)
        if debug:
            print(f"[create_figure] target_aspect={target_aspect:.12g}, achieved={ach:.12g}")
            print("width_ratios:", width_ratios)
            print("height_ratios:", height_ratios)

        return fig, gs

    fig, gs = create_figure()

    cm_ax0 = fig.add_subplot(gs[0, 1])  # top colormap axis
    cm_ax1 = fig.add_subplot(gs[1, 1], sharex=cm_ax0)  # bottom colormap axis
    cax = fig.add_subplot(gs[0:2, 2])

    ax_left_top = fig.add_subplot(gs[0, 0], sharey=cm_ax0)
    ax_left_bot = fig.add_subplot(gs[1, 0], sharey=cm_ax1, sharex=ax_left_top)

    ax_bottom = fig.add_subplot(gs[2, 1], sharex=cm_ax0)

    def enforce_space_lims():
        cm_ax0.set_xlim(x_min, x_max)
        ymax = y_space[-1]
        ymin = y_space[0]
        cm_ax0.set_ylim(ymax - distance + 0.5, ymax + 0.5)
        cm_ax1.set_ylim(ymin -0.5, ymin + distance - 0.5)

    def enforce_magn_lims(pad=0.17):
        max_x = np.max(magn_profile_x[int(x_min):int(x_max)])
        min_x = np.min(magn_profile_x[int(x_min):int(x_max)])
        range_x = max_x - min_x
        ax_bottom.set_ylim(min_x - range_x * pad, max_x + range_x * pad)
        max_y = np.max(magn_profile_y)
        min_y = np.min(magn_profile_y)
        range_y = max_y - min_y
        ax_left_top.set_xlim(min_y - range_y * pad, max_y + range_y * pad)

        return min_y - range_y * pad, max_y + range_y * pad, min_x - range_x * pad, max_x + range_x * pad

    def remove_spines():
        for ax in [ax_left_top, cm_ax0]:
            ax.spines.bottom.set_visible(False)
            ax.tick_params(bottom=False, labelbottom=False)
        for ax in [ax_left_bot, cm_ax1]:
            ax.spines.top.set_visible(False)
            ax.tick_params(top=False, labeltop=False)
        for ax in [cm_ax0, cm_ax1]:
            ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

        ax_bottom.tick_params(left=False, labelleft=False, right=True, labelright=True)
        ax_left_top.tick_params(rotation=-90)
        ax_left_bot.tick_params(rotation=-90)

    def magn_tick_labels():

        min_y, max_y, min_x, max_x = enforce_magn_lims()

        third_y = round((max_y - min_y) / 3, 4)
        y0 = - third_y
        y1 = third_y

        ax_left_bot.set_xticks([y0, 0.0, y1], labels=[str(y0), "", str(y1)])

        third_x = round((max_x - min_x) / 3, 4)
        x0 = - third_x
        x1 = third_x

        ax_bottom.set_yticks([x0, 0.0, x1], labels=[str(x0), "", str(x1)])

    def mark_Tstep():
        pos = helper.get_index_last_warm(0.49, 256) + 1.0   # because the profile is shifted by 0.5 to right
        for ax in [cm_ax0, cm_ax1, ax_bottom]:
            ax.axvline(pos, color="green", linewidth=1.0, linestyle="--", label=r"$\Delta T$")
        cm_ax0.legend(loc="lower right")

    def axes_labels(pad=0.055):
        ax_bottom.set_xlabel(x_space_label)
        ax_bottom.yaxis.set_label_position("right")
        ax_bottom.set_ylabel(magn_label)
        ax_left_bot.set_xlabel(magn_label)

        x_left = ax_left_top.get_position().get_points()[0][0]
        y1_left = ax_left_top.get_position().get_points()[0][1]
        y2_left = ax_left_bot.get_position().get_points()[1][1]
        fig.text(x_left - pad, (y1_left + y2_left) * 0.5, y_space_label, rotation=-90, va="center", ha="right")

    def broken_axis_markings():
        size = 6
        plot_util.add_axis_break_marking(ax_left_top, "bottom left", "vertical", size)
        plot_util.add_axis_break_marking(ax_left_top, "bottom right", "vertical", size)
        plot_util.add_axis_break_marking(ax_left_bot, "top left", "vertical", size)
        plot_util.add_axis_break_marking(ax_left_bot, "top right", "vertical", size)
        plot_util.add_axis_break_marking(cm_ax0, "bottom left", "vertical", size)
        plot_util.add_axis_break_marking(cm_ax0, "bottom right", "vertical", size)
        plot_util.add_axis_break_marking(cm_ax1, "top left", "vertical", size)
        plot_util.add_axis_break_marking(cm_ax1, "top right", "vertical", size)


    remove_spines()
    magn_tick_labels()
    enforce_space_lims()
    enforce_magn_lims()
    mark_Tstep()

    axes_labels()

    broken_axis_markings()

    cm_ax0.pcolormesh(X, Y, magnetization.T, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    im = cm_ax1.pcolormesh(X, Y, magnetization.T, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(magn_label)

    ax_bottom.plot(x_space, magn_profile_x, **profile_kwargs)
    ax_left_top.plot(magn_profile_y, y_space, **profile_kwargs)
    ax_left_bot.plot(magn_profile_y, y_space, **profile_kwargs)

    fig.savefig(f"{save_base_path}sne_spin_accu_config{save_suffix}.pdf")

    plt.show()


def spin_accu_config(tilted_data, direction):
    magnetization = spinconf_util.average_z_layers(physics.magnetization(
        spinconf_util.select_SL_and_component(tilted_data, "A", "z"),
        spinconf_util.select_SL_and_component(tilted_data, "B", "z")
    ), keepdims=False)

    magn_profile_x = np.mean(magnetization, axis=1)
    magn_profile_y = np.mean(magnetization, axis=0)

    warnings.warn("Taking profile average over whole region!")

    x_space = np.arange(0.5, magnetization.shape[0], 1)
    y_space = np.arange(0.5, magnetization.shape[1], 1)

    if direction == "-110":
        spin_accu_config_plot(magnetization.T, y_space, x_space, magn_profile_y, magn_profile_x,
                              x_space_label=r"position $x / \tilde{a}$ in direction \hkl[-110]",
                              y_space_label=r"position $y / \tilde{a}$ in direction \hkl[-1-10]",
                              save_suffix="_-110")
    elif direction == "110":
        spin_accu_config_plot(magnetization, x_space, y_space, magn_profile_x, magn_profile_y,
                              x_space_label=r"position $x / \tilde{a}$ in direction \hkl[110]",
                              y_space_label=r"position $y / \tilde{a}$ in direction \hkl[-110]",
                              save_suffix="_110")
    else:
        raise ValueError(f"Direction '{direction}' not valid.")


def sne_spin_accumulation():
    temperature = 2     # available: 1..10
    T = temperature

    paths = {
        "100": f"/data/scc/marian.gunsch/08/08_xTstep/T{T}/",
        "010": f"/data/scc/marian.gunsch/08/08_yTstep/T{T}/",
        "110": f"/data/scc/marian.gunsch/00/AM_tiltedX_Tstep_nernst/AM_tiltedX_Tstep_nernst_T{T}/",
        "-110": f"/data/scc/marian.gunsch/08/08_tilted_yTstep/T{T}/",
    }

    # for cold region
    equi_T0_paths = {
        "100": "/data/scc/marian.gunsch/12/AM_Tstairs_x_T0_openbou/",
        "010": "/data/scc/marian.gunsch/12/AM_Tstairs_y_T0_openbou/",
        "110": "/data/scc/marian.gunsch/12/AM_tilt_Tstairs_x_T0_openbou/",
        "-110": "/data/scc/marian.gunsch/12/AM_tilt_Tstairs_y_T0_openbou/"
    }

    is_tilted = {
        "100": False,
        "010": False,
        "110": True,
        "-110": True
    }

    directions = paths.keys()

    conf_data = spinconf_util.npy_file_from_dict(paths, is_tilted)

    # Config plots: To get an idea of what the profile plots later on show
    spin_accu_config(conf_data["-110"], "-110")
    spin_accu_config(conf_data["110"], "110")

    # Profile plots
    # TODO: Take appropriate averages over config data! Plot that for all available directions and compare somehow
    # TODO: Maybe compare to ground state? (ground state being T=0)
    # TODO: Maybe also a comparison like 50 points away from T step? To confirm it has to be some perpendicular effect?
    # TODO: idk
    warnings.warn("Stuff todo!")



# %% Spin currents (transversal)

def spin_currents_open():
    # measurement direction (profile axis)
    paths = {
        "100": "/data/scc/marian.gunsch/11/AM_xTstep_y/",
        "010": "/data/scc/marian.gunsch/11/AM_yTstep_x/",
        "-110": "/data/scc/marian.gunsch/11/AM_tilt_xTstep_y/",
        "110": "/data/scc/marian.gunsch/11/AM_tilt_yTstep_x/"
    }
    # Direction of the temperature step
    step_dir = {
        "010": "100",
        "100": "010",
        "-110": "110",
        "110": "-110"
    }

    mag_util.npy_files_from_dict(paths)


def spin_currents_upperABC():
    pass
    print("Massive problems running simulations.")
    print("Jobscripts labelled with '11' ")


def spin_currents_uploABC():
    pass
    print("Massive problems running simulations.")
    print("Jobscripts labelled with '11' ")


# %% Magnon spectrum

def sne_magnon_spectrum():
    # direction of measurement (profile axis)
    paths = {
        "-110": "/data/scc/marian.gunsch/02/02_AM_tilted_Tstep_hightres/",
        "110": "/data/scc/marian.gunsch/14/AM_tilt_yTstep_x_hightres/"
    }
    # Direction of the temperature step
    step_dir = {
        "-110": "110",
        "110": "-110"
    }

    mag_util.npy_files_from_dict(paths)


# %% Main

def main():
    sne_spin_accumulation()
    # spin_currents_open()
    # spin_currents_upperABC()
    # spin_currents_uploABC()
    # sne_magnon_spectrum()
