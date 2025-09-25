import warnings

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.colors as colors
from matplotlib.collections import PolyCollection
from scipy.optimize import curve_fit

import src.utility as util
import src.mag_util as mag_util
import src.bulk_util as bulk_util
import src.utility as util
import src.plot_util as plot_util
import src.physics as physics
import src.spinconf_util as spinconf_util
import src.helper as helper
import src.save_util as save_util

# %%
save_base_path = "out/thesis/sse/"


shared_kwargs = dict(markersize=0.5, linewidth=0.8)
plot_kwargs_dict = {
    "1-10": dict(**shared_kwargs, marker="", linestyle="-"),     # marker="s"
    "110": dict(**shared_kwargs, marker="", linestyle="--"),     # marker="D"
    "100": dict(**shared_kwargs, marker="", linestyle=":")
}

paths = {
    "1-10": dict(),
    "110": dict()
}

for B in range(50, 100 + 1, 10):        # 50, 60, ..., 90, 100
    paths["1-10"][B] = f"/data/scc/marian.gunsch/05/05_AM_tilted_xTstep_T2_B{B}/"
    paths["110"][B] = f"/data/scc/marian.gunsch/05/05_AM_tilted_yTstep_T2_B{B}/"

for B_ in range(50, 90 + 1, 10):       # -50, -60, ..., -90
    B = -B_
    paths["1-10"][B] = f"/data/scc/marian.gunsch/17/AM_tilt_xTstep_T2_B/n{B_}/"
    paths["110"][B] = f"/data/scc/marian.gunsch/17/AM_tilt_yTstep_T2_B/n{B_}/"

paths["1-10"][0] = "/data/scc/marian.gunsch/04/04_AM_tilted_xTstep_T2-2/"
paths["110"][0] = "/data/scc/marian.gunsch/04/04_AM_tilted_yTstep_T2-2/"
paths["1-10"][-100] = "/data/scc/marian.gunsch/06/06_AM_tilted_xTstep_T2_Bn100/"
paths["110"][-100] = "/data/scc/marian.gunsch/06/06_AM_tilted_yTstep_T2_Bn100/"

B_fields = paths["1-10"].keys()
if B_fields != paths["110"].keys():
    raise AttributeError("Conflicting keys!")

directions = paths.keys()
field_strengths = sorted(paths["1-10"].keys(), reverse=False)
assert field_strengths == sorted(paths["110"].keys(), reverse=False)

data_A = {
    "1-10": dict(),
    "110": dict()
}
data_B = {
    "1-10": dict(),
    "110": dict()
}

magnetization = {
    "1-10": dict(),
    "110": dict()
}

def initialize_data():
    data_A["1-10"], data_B["1-10"] = mag_util.npy_files_from_dict(paths["1-10"])
    data_A["110"], data_B["110"] = mag_util.npy_files_from_dict(paths["110"])
    for direction in directions:
        for B in field_strengths:
            magnetization[direction][B] = physics.magnetization(
                mag_util.get_component(data_A[direction][B], "z", 0),
                mag_util.get_component(data_B[direction][B], "z", 0), True)


# %% INTRODUCTION OF A STATIC MAGNETIC FIELD (MAGNETIZATION)

def plot_magn_Bfield(magn_dict, ylabel=r"magnetization $\langle S^z \rangle$", xlim=(0, 255),
                     field_strengths_=(-100, 0, 50, 60, 70, 80, 90, 100)):

    if len(field_strengths_) > 8:
        warnings.warn(f"{len(field_strengths_)} field strengths are a lot to plot... Messy, huh?")

    fig, ax = plt.subplots()
    ax.axhline(0, color="gray", linestyle="-", marker="", linewidth=0.5)

    max_val = - np.inf
    min_val = np.inf

    colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']

    lines = dict((direction, []) for direction in directions)

    for direction in directions:
        for B, color in zip(field_strengths_, colors):
            # label = r"$\num{" + str(B) + r"}{}$" if direction == "1-10" else None
            label = r"$\num{" + str(B) + r"}{}$"
            _line, = ax.plot(magn_dict[direction][B], **plot_kwargs_dict[direction],
                            label=label, color=color)
            lines[direction].append(_line)

            max_val = max(max_val, np.max(magn_dict[direction][B][15:-15]))
            min_val = min(min_val, np.min(magn_dict[direction][B][15:-15]))

    val_range = max_val - min_val
    pad = 0.05

    ax.set_xlabel(r"position $x/\tilde{a}$")
    ax.set_ylabel(ylabel)

    ax.set_xlim(*xlim)
    ax.set_ylim(min_val - pad * val_range, max_val + pad * val_range)

    legend1 = plt.legend(handles=lines["1-10"], ncols=2, title=f"\\hkl[1-10]: " + r"$B$ (\si{\tesla})",
                         fontsize="small", loc="upper right")
    legend2 = plt.legend(handles=lines["110"], ncols=2, title=f"\\hkl[110]: " + r"$B$ (\si{\tesla})",
                         fontsize="small", loc="lower right")
    ax.add_artist(legend1)
    ax.add_artist(legend2)

    dT_pt = plot_util.place_Tstep_marking(ax, helper.get_actual_Tstep_pos(0.49, 256))
    # dT_pt, = ax.plot(helper.get_actual_Tstep_pos(0.49, 256), min_val - pad * val_range, marker=10,
    #                  color="r", linestyle="", label=r"$\Delta T$")
    # ax.plot(helper.get_actual_Tstep_pos(0.49, 256), max_val + pad * val_range, marker=11, color="r",
    #         linestyle="", )

    legend_dT = plt.legend(handles=[dT_pt,], loc="lower left")
    ax.add_artist(legend_dT)

    fig.tight_layout()

    # ax.set_xticks(list(ax.get_xticks()) + [helper.get_Tstep_pos(0.49, 256),],
    #               list(ax.get_xticklabels()) + ["$\Delta T$",])


    return fig, ax



# %% magnetization for different magnetic fields
def sse_magnetization_Bfield():
    fig, ax = plot_magn_Bfield(magnetization)

    fig.savefig(f"{save_base_path}magnetization_Bfield.pdf")
    plt.show()



# %% magnon accumulation for different magnetic fields
def sse_magnaccum_Bfield():

    equi_slice = slice(9, 40)

    equilibrium_magn = dict()

    for B in field_strengths:
        l = []
        for direction in directions:
            l.append(np.mean(magnetization[direction][B][equi_slice]))
        equilibrium_magn[B] = np.mean(np.array(l))

    step_pos = helper.get_absolute_T_step_index(0.49, 256)

    magnon_accumulation = {
        "1-10": dict(),
        "110": dict()
    }

    for direction in directions:
        for B in field_strengths:
            magnon_accumulation[direction][B] = np.copy(magnetization[direction][B])
            magnon_accumulation[direction][B][:step_pos] -= equilibrium_magn[B]

    fig, ax = plot_magn_Bfield(magnon_accumulation, r"magnon accum. $\langle \Delta S^z \rangle$", (60, 215))
    fig.savefig(f"{save_base_path}magnonaccum_Bfield.pdf")
    plt.show()

    fig, ax = plot_magn_Bfield(magnon_accumulation, r"magnon accum. $\langle \Delta S^z \rangle$", (115, 135))
    fig.savefig(f"{save_base_path}magnonaccum_Bfield_zoom__.pdf")
    plt.show()


# %% Plot peaks vs magnetic field
def peak_dependence():
    area_slice = slice(helper.get_index_first_cold(0.49, 256), None)

    abs_max_func = {
        "1-10": np.max,
        "110": np.min
    }
    peaks = {
        "1-10": dict(),
        "110": dict()
    }

    for direction in directions:
        for B in field_strengths:
            peaks[direction][B] = abs_max_func[direction](magnetization[direction][B][area_slice])


    def linear_func(x, m, c):
        x = np.array(x)
        return m * x + c

    popt = dict()
    pcov = dict()
    perr = dict()
    for direction in directions:
        popt[direction], pcov[direction] = curve_fit(linear_func, field_strengths, [peaks[direction][B] for B in field_strengths])
        perr[direction] = np.sqrt(np.diag(pcov[direction]))

        print(f"{popt=}")
        print(f"{pcov=}")
        print(f"{perr=}")

    # Plot

    fig, ax = plt.subplots()

    ax.set_xlabel(r"$B$ (\si{\tesla})")
    ax.set_ylabel(r"peak magnon accum. $ \langle \Delta S^z \rangle_{\max}$")

    sign = + 1

    ax.plot(field_strengths, linear_func(field_strengths, *popt["1-10"]), linestyle="-", marker="", color="k")
    ax.plot(field_strengths, sign * linear_func(field_strengths, *popt["110"]), linestyle="-", marker="", color="k")

    ax.plot(field_strengths, [peaks["1-10"][B] for B in field_strengths], label=r"\hkl[1-10]", linestyle="", marker="o")
    ax.plot(field_strengths, [sign * peaks["110"][B] for B in field_strengths], label=r"\hkl[110]", linestyle="", marker="s")

    ax.legend(loc="upper left")

    fig.tight_layout()

    fig.savefig(f"{save_base_path}sse_peak_dependence.pdf")

    plt.show()



# %% Compare for different directions for one magnetic field strength: magnetization for [100], [1-10], [110]
def direction_comparison():
    path_100 = "/data/scc/marian.gunsch/13/13_AM_xTstep_T2_B100/"
    # path_100 = "/data/scc/marian.gunsch/13/13_AM_xTstep_T2_Bn100/"
    B_strength = +100

    eq_path = "/data/scc/marian.gunsch/10/AM_Tstairs_T2_x_B100-2/"
    # eq_path = "/data/scc/marian.gunsch/06/06_AM_tilted_xTstep_T2_Bn100/"
    tempA, tempB = mag_util.npy_files(eq_path)
    eq_magn = np.mean(physics.magnetization(
        mag_util.get_component(tempA, "z", ),
        mag_util.get_component(tempB, "z", ), True)
    )

    step_pos = helper.get_absolute_T_step_index(0.49, 256)
    max_magn = - np.inf
    max_accu = - np.inf
    min_magn = np.inf
    min_accu = np.inf
    pad = 0.05

    tempA, tempB = mag_util.npy_files(path_100)
    data_A_ = {
        "100": tempA,
        "1-10": data_A["1-10"][B_strength],
        "110": data_A["110"][B_strength]
    }
    data_B_ = {
        "100": tempB,
        "1-10": data_B["1-10"][B_strength],
        "110": data_B["110"][B_strength]
    }

    lattice_const = {
        "100": physics.lattice_constant,
        "1-10": physics.lattice_constant_tilted,
        "110": physics.lattice_constant_tilted
    }

    directions_ = data_A_.keys()
    magnetization_ = dict()
    magn_accumulation = dict()

    for direction in directions_:
        magnetization_[direction] = physics.magnetization(
            mag_util.get_component(data_A_[direction], "z", ),
            mag_util.get_component(data_B_[direction], "z", ), True)

        magn_accumulation[direction] = np.copy(magnetization_[direction])
        magn_accumulation[direction][:step_pos] -= eq_magn

        max_magn = max(max_magn, np.max(magnetization_[direction][15:-15]))
        max_accu = max(max_accu, np.max(magn_accumulation[direction][15:-15]))
        min_magn = min(min_magn, np.min(magnetization_[direction][15:-15]))
        min_accu = min(min_accu, np.min(magn_accumulation[direction][15:-15]))


    magn_range = max_magn - min_magn
    accu_range = max_accu - min_accu

    def plot():
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

        ax2.set_xlabel("$x / a$")
        ax1.set_ylabel(r"$\langle S^z \rangle$")
        ax2.set_ylabel(r"$\langle \Delta S^z \rangle$")

        ax2.axhline(0, color="gray", linestyle="-", marker="", linewidth=0.5)

        actual_step_pos = helper.get_actual_Tstep_pos(0.49, 256)
        ax1.set_xlim(- 60, + 60)
        ax1.set_ylim(min_magn - pad * magn_range, max_magn + pad * magn_range)
        ax2.set_ylim(min_accu - pad * accu_range, max_accu + pad * accu_range)

        custom_plot_kwargs = plot_kwargs_dict.copy()
        custom_plot_kwargs["1-10"].update(dict(linestyle="-", marker="s", markersize=1.2))
        custom_plot_kwargs["110"].update(dict(linestyle="-", marker="D", markersize=1.2))
        custom_plot_kwargs["100"].update(dict(linestyle="-", marker="o", markersize=1.2))

        for direction in directions_:
            x = (np.arange(0, magnetization_[direction].shape[0], dtype=float) - actual_step_pos) * lattice_const[direction]
            ax1.plot(x, magnetization_[direction], label=fr"\hkl[{direction}]", **custom_plot_kwargs[direction])

            ax2.plot(x, magn_accumulation[direction], label=fr"\hkl[{direction}]", **custom_plot_kwargs[direction])

        # dT_pt = None
        # for ax, min_val, max_val in zip((ax1, ax2), (min_magn, min_accu), (max_magn, max_accu)):
        #     val_range = max_val - min_val
        #     label = r"$\Delta T$" if ax == ax2 else None
        #     dT_pt, = ax.plot(0, min_val - pad * val_range, marker=10,
        #                      color="r", linestyle="", label=label)
        #     ax.plot(0, max_val + pad * val_range, marker=11, color="r",
        #             linestyle="",)


        plot_util.place_Tstep_marking(ax1, 0, None)
        dT_pt = plot_util.place_Tstep_marking(ax2, 0)

        legend_dT = plt.legend(handles=[dT_pt,], loc="lower left")
        ax2.add_artist(legend_dT)

        ax1.legend()
        fig.tight_layout()

        fig.savefig(f"{save_base_path}direction_comparison.pdf")

        plt.show()

    plot()



# %% Propagation lengths for [1-10] and [110] for different B-fields
def propagation_lengths():
    x0 = 3
    start = helper.get_index_first_cold(0.49, 256) + x0
    fit_area = slice(start, start + 45)

    def exp_func1(x, A, l):
        x = np.array(x)
        return A * np.exp(-x / l)

    def exp_func5(x, A_beta, lambda_beta, A_alpha, lambda_alpha):
        x = np.array(x)
        return A_beta * np.exp(-x / lambda_beta) - A_alpha * np.exp(-x / lambda_alpha)

    def exp_func6(x, A_beta, lambda_beta, A_alpha, lambda_alpha):
        x = np.array(x)
        return A_beta * np.exp(-(x - x0) / lambda_beta) - A_alpha * np.exp(-(x - x0) / lambda_alpha)


    popt = dict()
    pcov = dict()
    perr = dict()

    def attempt(fit_func, lower, upper, p0, log=False):
        for direction in directions:
            popt[direction] = dict()
            pcov[direction] = dict()
            perr[direction] = dict()

            for B in field_strengths:
                popt[direction][B], pcov[direction][B] = curve_fit(
                    fit_func, np.arange(fit_area.stop - fit_area.start), magnetization[direction][B][fit_area],
                    bounds=(lower, upper), p0=p0
                )
                perr[direction][B] = np.sqrt(np.diag(pcov[direction][B]))

        if log:
            print(f"popt:\n{popt}\n\n"
                  f"pcov:\n{pcov}\n\n"
                  f"perr:\n{perr}\n"
                  )
        return fit_func

    def biexp_attempt():
        fit_func = exp_func6

        p0 = (0.995, 50, 0.995, 50)
        lower = (0.8, 3, 0.8, 3)
        upper = (1.0, 100, 1.0, 100)

        return attempt(fit_func, lower, upper, p0, log=True)

    def mono_exp(log=False):
        fit_func = exp_func1

        p0 = (0.002, 50)
        lower = (-0.1, 3)
        upper = (0.1, 100)

        return attempt(fit_func, lower, upper, p0, log=log)


    fit_func = mono_exp()

    def plot():
        x_axis = np.arange(len(magnetization["1-10"][0]))
        x_fit = np.linspace(0, fit_area.stop - fit_area.start, 300)

        fig, ax = plt.subplots()

        print(field_strengths)
        for direction in directions:
            for B in field_strengths:
                ax.plot(x_axis, magnetization[direction][B], linestyle="", marker="o", markersize=0.2,
                        label=f"[{direction}], {B=}")
                ax.plot(x_fit + fit_area.start, fit_func(x_fit, *popt[direction][B]), linestyle="-", marker="",
                        linewidth=0.2, color="k")
                ax.legend(loc="upper right", fontsize="x-small")

        ax.set_xlim(120, 200)
        ax.set_ylim(-0.002, 0.002)

        plt.show()

    plot()

    if fit_func != exp_func1:
        warnings.warn("I have only implemented a mono-exponential fit.")
        return

    # Plotting...

    fig, ax = plt.subplots()

    ax.set_xlabel(r"$B$ (\si{\tesla})")
    ax.set_ylabel(r"propagation length $\lambda / \tilde{a}$")

    for direction in directions:
        length = [popt[direction][B][1] for B in field_strengths]
        length_unc = [perr[direction][B][1] for B in field_strengths]
        ax.errorbar(field_strengths, length, yerr=length_unc, marker="_", capsize=2.7, label=rf"\hkl[{direction}]",
                    # linestyle=(0, (5, 10)), linewidth=0.5
                    linestyle=""
                    )
        print(f"x: {field_strengths}")
        print(f"y: {np.array(length)}")

    ax.legend(loc="lower center")

    fig.tight_layout()

    fig.savefig(f"{save_base_path}propagation_length_dependence.pdf")

    plt.show()



# %% SPIN CURRENTS
def sse_spin_currents(from_profile=True, xlim=(-111.5, 141.5)):

    normed_units = True

    warnings.warn("Use updated paths, once simulations are finished running.")
    paths = {       # if starts running: -2 with 128 z layers to reduce noise
        "100": "/data/scc/marian.gunsch/16/AM_xTstep_T2-2/",        # with or without _noABC? (makes no difference)
        "010": "/data/scc/marian.gunsch/16/AM_yTstep_T2-2/",          # Not sure whether to use at all
        "1-10": "/data/scc/marian.gunsch/04/04_AM_tilted_xTstep_T2-2/",
        "110": "/data/scc/marian.gunsch/04/04_AM_tilted_yTstep_T2-2/"
    }

    if from_profile:
        dataA_, dataB_ = mag_util.npy_files_from_dict(paths)
    else:
        curr_dir = { "100": "x", "1-10": "x", "010": "y", "110":"y"}
        conf_data = spinconf_util.npy_file_from_dict(paths)

    shift = 0.5     # this is needed because we are working with currents (in between two layers/atoms)

    directions_ = paths.keys()

    currents = dict()
    N = dict()

    for direction in directions_:
        if from_profile:
            currents[direction] = mag_util.spin_current(dataA_[direction], dataB_[direction], direction, normed_units=normed_units)
            N = mag_util.get_component(dataA_[direction], "x").shape[1]
        else:
            currents[direction] = spinconf_util.spin_current(conf_data[direction], curr_dir[direction], direction, curr_dir[direction],
                                                             normed_units=normed_units)
            N[direction] = conf_data[direction].shape[0] if curr_dir[direction] == "x" else conf_data[direction].shape[1]

    Tstep_pos = dict()
    x_space = dict()
    for direction in directions_:
        Tstep_pos[direction] = helper.get_actual_Tstep_pos(0.49, N[direction])
        x = np.arange(currents[direction].shape[0]) + shift
        x -= Tstep_pos[direction]
        x = physics.index_to_position(x, direction)
        x_space[direction] = x

    fig, ax = plt.subplots()
    ax.set_xlabel("position $x/a$")
    # ax.set_ylabel(r"spin current $j^{\mathrm{L}}$ (\si{\meter\per\second})")
    ax.set_ylabel(r"spin current $j^{\mathrm{L}}$ ($\tfrac{\gamma_{\mathrm{e}} \, a \, J_1}{\mu_{\mathrm{B}}}$)")
    ax.set_xlim(*xlim)

    lines = []

    for direction in directions_:
        # line, = ax.plot(x_tilted, currents[direction], label=fr"\hkl[{direction}]")
        line, = ax.plot(x_space[direction], currents[direction], label=fr"\hkl[{direction}]")
        lines.append(line)

    legend = plt.legend(handles=lines, loc="upper right")
    ax.add_artist(legend)

    handle = plot_util.place_Tstep_marking(ax, 0.0)
    legend_dT = plt.legend(handles=[handle, ], loc="lower left")
    ax.add_artist(legend_dT)

    fig.tight_layout()

    suffix = "_fromprof" if from_profile else "_fromconf"
    save_path = f"{save_base_path}sse_spin_currents{suffix}.pdf"
    fig.savefig(save_path)
    save_util.source_paths(save_path, paths)
    shared_description = ("Longitudinal spin current for crystallographic directions. x values for 1-10 and 110 have "
                          "been corrected for the appropriate lattice constant.")
    if from_profile:
        save_util.description(save_path, f"Used profile data to calculate spin current. {shared_description}")
    else:
        save_util.description(save_path, f"Used config data to calculate spin current. {shared_description}")

    plt.show()


def sse_spin_currents_comparison(xlim=(-101.5, 126.5)):
    normed_units = True

    paths = {
        False : {
            "100": "/data/scc/marian.gunsch/16/AM_xTstep_T2-2/",
            "010": "/data/scc/marian.gunsch/16/AM_yTstep_T2-2/",
            "1-10": "/data/scc/marian.gunsch/04/04_AM_tilted_xTstep_T2-2/",
            "110": "/data/scc/marian.gunsch/04/04_AM_tilted_yTstep_T2-2/"
        },
        True : {
            "100": "/data/scc/marian.gunsch/13/13_AM_xTstep_T2_B100/",
            "010": "/data/scc/marian.gunsch/13/13_AM_yTstep_T2_B100/",
            "1-10": "/data/scc/marian.gunsch/05/05_AM_tilted_xTstep_T2_B100/",
            "110": "/data/scc/marian.gunsch/05/05_AM_tilted_yTstep_T2_B100/"
        }
    }

    shift = 0.5

    field_strengths = paths.keys()
    directions = paths[False].keys()
    curr_dir = {"100": "x", "1-10": "x", "010": "y", "110": "y"}

    conf_data = {B: spinconf_util.npy_file_from_dict(paths[B]) for B in field_strengths}

    currents = dict()
    x_space = dict()

    for B in field_strengths:
        currents[B] = dict()
        x_space[B] = dict()
        for d in directions:
            currents[B][d] = spinconf_util.spin_current(conf_data[B][d] , curr_dir[d], d,
                                                             curr_dir[d],
                                                             normed_units=normed_units)
            N = conf_data[B][d].shape[0] if curr_dir[d] == "x" else conf_data[B][d] .shape[1]

            Tstep_pos = helper.get_actual_Tstep_pos(0.49, N)
            x = np.arange(currents[B][d].shape[0]) + shift
            x -= Tstep_pos
            x = physics.index_to_position(x, d)
            x_space[B][d] = x


    # Plotting

    fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)
    axs.flatten()
    axs_dict = { False: axs[0], True: axs[1] }
    axs[0].set_xlabel("$x/a$")
    axs[1].set_xlabel("$x/a$")
    axs[0].axhline(0, color="gray", linestyle="-", marker="", linewidth=0.5)
    axs[1].axhline(0, color="gray", linestyle="-", marker="", linewidth=0.5)

    if normed_units:
        axs[0].set_ylabel(r"spin current $j^{\mathrm{L}}$ ($\tfrac{\gamma_{\mathrm{e}} \, a \, J_1}{\mu_{\mathrm{B}}}$)")
    else:
        axs[0].set_ylabel(r"spin current $j^{\mathrm{L}}$ (\si{\meter\per\second})")
    axs[0].set_xlim(*xlim)

    axs_dict[False].set_title(r"$B = 0$")
    axs_dict[True].set_title(r"$B > 0$")

    lines = { False: [], True: []}

    for B in field_strengths:
        for d in directions:
            line, = axs_dict[B].plot(x_space[B][d], currents[B][d], label=fr"\hkl[{d}]")
            lines[B].append(line)

    legend = axs_dict[True].legend(handles=lines[True], loc="upper right")
    axs_dict[True].add_artist(legend)

    plot_util.place_Tstep_marking(axs[1], 0.0)
    handle = plot_util.place_Tstep_marking(axs[0], 0.0)
    legend_dT = axs[0].legend(handles=[handle, ], loc="lower left")
    axs[0].add_artist(legend_dT)

    fig.tight_layout()

    save_path = f"{save_base_path}sse_spin_currents_comparison.pdf"
    fig.savefig(save_path)
    save_util.source_paths(save_path, paths)
    shared_description = ("Longitudinal spin current for crystallographic directions. x values for 1-10 and 110 have "
                          "been corrected for the appropriate lattice constant.")
    save_util.description(save_path, f"Used config data to calculate spin current. {shared_description}")

    plt.show()


# %% Main

def main():
    initialize_data()

    sse_magnetization_Bfield()
    sse_magnaccum_Bfield()

    peak_dependence()

    direction_comparison()

    propagation_lengths()

    sse_spin_currents(False)
    sse_spin_currents_comparison()

