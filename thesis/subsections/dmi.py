import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, FancyArrowPatch, Circle
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.colors as colors
from matplotlib.collections import PolyCollection
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D

import warnings

import src.utility as util
import src.mag_util as mag_util
import src.bulk_util as bulk_util
import src.utility as util
import src.plot_util as plot_util
import src.physics as physics
import src.spinconf_util as spinconf_util
import src.helper as helper
import src.save_util as save_util
import thesis.mpl_configuration as mpl_config

# %%

save_base_path = "out/thesis/dmi/"


# %% EQUILIBRIUM PROPERTIES

def equilibrium_comparison_plot(S, magn, tick_labels):
    plot_kwargs = dict(marker="o", linestyle="")

    components = ["x", "y", "z"]
    sublattices = ["A", "B"]
    DMI = [False, True]
    x_positions = {False: 0.6, True: 2.4}
    xlim = (0, 3)

    fig = plt.figure()
    gs = fig.add_gridspec(nrows=3, ncols=4, height_ratios=(2, 2, 2), width_ratios=(3, 3, 0.9, 3),
                          hspace=0.1,
                          left=0.15, right=0.9, bottom=0.15, top=0.9)

    axs = dict()
    ref_ax = fig.add_subplot(gs[0, 0])


    axs["x"] = (ref_ax, fig.add_subplot(gs[1, 0], sharex=ref_ax), fig.add_subplot(gs[2, 0], sharex=ref_ax))
    axs["y"] = (fig.add_subplot(gs[0, 1], sharex=ref_ax, sharey=axs["x"][0]),
                fig.add_subplot(gs[1, 1], sharex=ref_ax, sharey=axs["x"][1]),
                fig.add_subplot(gs[2, 1], sharex=ref_ax, sharey=axs["x"][2]))
    axs["z"] = (fig.add_subplot(gs[0, 3], sharex=ref_ax, sharey=axs["x"][0]),
                fig.add_subplot(gs[1, 3], sharex=ref_ax),
                fig.add_subplot(gs[2, 3], sharex=ref_ax))

    sign_B = +1
    for c in components:
        for dmi in DMI:
            axs[c][0].plot(x_positions[dmi], magn[dmi][c], **plot_kwargs)
            axs[c][1].plot(x_positions[dmi], S[dmi]["A"][c], **plot_kwargs)
            axs[c][2].plot(x_positions[dmi], sign_B * S[dmi]["B"][c], **plot_kwargs)

    def set_lims(pad=0.1):
        ref_ax.set_xlim(*xlim)

        factor = 2/2
        axs["x"][0].margins(y=pad)
        axs["x"][1].margins(y=pad * factor)       # 2/1.1=1.81
        axs["z"][1].margins(y=pad * factor)
        axs["x"][2].margins(y=pad * factor)
        axs["z"][2].margins(y=pad * factor)

    def handle_tick_labels():
        for i in range(3):
            axs["y"][i].tick_params(labelleft=False)
        # axs["z"][0].tick_params(labelleft=False)
        for c in components:
            for i in range(2):
                axs[c][i].tick_params(labelbottom=False)

        ref_ax.set_xticks(list(x_positions.values()), list(tick_labels.values()))    # a bit dangerous
        axs["z"][1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        axs["z"][2].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    set_lims()
    handle_tick_labels()

    axs["x"][0].set_ylabel(r"$\langle S^{\eta}_{\mathrm{net}} \rangle$")
    axs["x"][1].set_ylabel(r"$\langle S^{\eta}_{\mathrm{B}} \rangle$")
    axs["x"][2].set_ylabel(r"$\langle S^{\eta}_{\mathrm{B}} \rangle$")
    for c in components:
        axs[c][0].set_title(rf"$\eta = {c}$")

    save_path = f"{save_base_path}equi_dmi_components.pdf"
    fig.savefig(save_path)

    plt.show()

    return save_path


def fancy_equilibrium_comparison_plot(S, magn, titles):
    arrow_kwargs = dict(arrowstyle="-|>", mutation_scale=20, shrinkA=0, shrinkB=0, lw=1.5)

    DMI = [False, True]

    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=mpl_config.get_size(0.8))
    axs = axs.flatten()

    for i, dmi in enumerate(DMI):
        axs[i].axhline(-1.0, alpha=0.8, color="grey", linestyle="--", linewidth=0.6)
        axs[i].axhline(1.0, alpha=0.8, color="grey", linestyle="--", linewidth=0.6)

        start = (0, 0)

        endA = (S[dmi]["A"]["y"], S[dmi]["A"]["z"])
        endB = (S[dmi]["B"]["y"], S[dmi]["B"]["z"])
        endNet = (magn[dmi]["y"], magn[dmi]["z"])

        arrowA = FancyArrowPatch(start, endA, **arrow_kwargs, color="blue")
        arrowB = FancyArrowPatch(start, endB, **arrow_kwargs, color="red")
        arrow_net = FancyArrowPatch(start, endNet, **arrow_kwargs, color="black")

        point_style = dict(marker="o", markeredgecolor="white", markersize=1.8, linestyle="", linewidth=0.03)

        axs[i].add_patch(arrowA)
        axs[i].plot(*endA, color="blue", **point_style)
        axs[i].add_patch(arrowB)
        axs[i].plot(*endB, color="red", **point_style)
        if dmi:
            axs[i].add_patch(arrow_net)
            axs[i].plot(*endNet, color="k", **point_style)

        # --- Add text labels with slight offsets ---
        label_ = lambda text : r"$\langle \vec{S}_{\mathrm{" + text + r"}} \rangle$"
        axs[i].text(endA[0] + 0.04, 0.9 * endA[1], label_("A"), color="blue", va="center", ha="left")
        axs[i].text(endB[0] + 0.04, 0.9 * endB[1], label_("B"), color="red", va="center", ha="left")
        if dmi:
            axs[i].text(endNet[0], endNet[1] + 0.06, label_("net"), color="black", va="bottom", ha="center")

        axs[i].set_title(list(titles.values())[i])

        # circle = Circle(start, radius=1, fill=False, alpha=0.8, color="grey", linestyle="--")
        # axs[i].add_patch(circle)


    axs[0].set_xlim(-0.19, 0.19)
    axs[0].set_ylim(-1.1, 1.1)

    axs[0].set_xlabel(r"$\langle S^y \rangle$")
    axs[1].set_xlabel(r"$\langle S^y \rangle$")
    axs[0].set_ylabel(r"$\langle S^z \rangle$")

    fig.tight_layout()

    save_path = f"{save_base_path}equi_dmi_components_fancy.pdf"
    fig.savefig(save_path)

    plt.show()

    return save_path


def average_spin_components():
    paths = {
        False: "/data/scc/marian.gunsch/10/AM_tilt_Tstairs_T2_x-2/",
        True: "/data/scc/marian.gunsch/00/AM_tiltedX_ttmstairs_DMI_ferri-2/"
    }
    labels = {
        False: "no DMI",
        True: "DMI"
    }

    dataA, dataB = mag_util.npy_files_from_dict(paths)
    data = dict(A=dataA, B=dataB)

    components = ["x", "y", "z"]
    sublattices = ["A", "B"]
    DMI = [False, True]

    S = dict()
    magn = dict()
    for dmi in DMI:
        S[dmi] = dict()
        magn[dmi] = dict()
        for sl in sublattices:
            S[dmi][sl] = dict()
            for component in components:
                S[dmi][sl][component] = np.mean(mag_util.get_component(data[sl][dmi], component))
        for component in components:
            magn[dmi][component] = physics.magnetization(S[dmi]["A"][component], S[dmi]["B"][component])

    # S[dmi][sl][component]
    # magn[dmi][component]

    # save_path = equilibrium_comparison_plot(S, magn, labels)
    # save_util.source_paths(save_path, paths)

    save_path = fancy_equilibrium_comparison_plot(S, magn, labels)
    save_util.source_paths(save_path, paths)

    # Think about what exactly to compare? Dmi/no DMI for different temperatures? Or just one temperature?


def dispersion_relation_dmi():
    # TODO!
    paths = {
        "100": "/data/scc/marian.gunsch/15/AM_DMI_Tstairs_T2_x/",     # not sure whether to use
        "010": "/data/scc/marian.gunsch/15/AM_DMI_Tstairs_T2_y/",     # not sure whether to use
        "110": "/data/scc/marian.gunsch/15/AM_tilt_DMI_Tstairs_T2_x/",
        "-110": "/data/scc/marian.gunsch/15/AM_tilt_DMI_Tstairs_T2_y/"
    }

    paths_nodmi = {
        "100": f"/data/scc/marian.gunsch/10/AM_Tstairs_T2_x-2/",      # not sure whether to use
        "010": f"/data/scc/marian.gunsch/10/AM_Tstairs_T2_y-2/",      # not sure whether to use
        "110": f"/data/scc/marian.gunsch/10/AM_tilt_Tstairs_T2_x-2/",
        "-110": f"/data/scc/marian.gunsch/10/AM__tilt_Tstairs_T2_y-2/"  # oups: '__'
    }

    directions = paths.keys()
    if directions != paths.keys():
        raise AttributeError("Conflicting keys!")

    from thesis.subsections.equilibrium import dispersion_comparison_table_data, dispersion_comparison_table_plot

    k_dict, freq_dict, magnon_density_dict = dispersion_comparison_table_data(paths_nodmi, paths)
    dispersion_comparison_table_plot(k_dict, freq_dict, magnon_density_dict, version=2,
                                     left_title="no DMI", right_title="DMI",
                                     save_path=f"{save_base_path}dispersion_comparison_dmi_table.pdf")


# %% SPIN SEEBECK EFFECT

def plot_sse(x_space, magn_dict, xlim=(-76, 76), ylim=(-0.0024, 0.0014), save_name=None):

    DMI = [False, True]
    components = ["y", "z"]
    directions = magn_dict[False].keys()

    plot_kwargs_dict = { False: dict(linestyle="--", linewidth=0.7),
                         True: dict(linestyle="-", linewidth=0.7) }

    fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)
    axs = axs.flatten()
    axs = dict(y=axs[0], z=axs[1])

    colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']

    lines = dict((c, dict((dmi, []) for dmi in DMI)) for c in components)

    for dmi in DMI:
        for d in directions:
            for c in components:
                _line, = axs[c].plot(x_space[dmi][d], magn_dict[dmi][d][c], label=rf"\hkl[{d}]",
                                       **plot_kwargs_dict[dmi])
                lines[c][dmi].append(_line)

    for c in components:
        axs[c].set_xlim(*xlim)
        axs[c].set_ylim(*ylim)
        axs[c].set_ylabel(rf"$\langle \Delta m^{c} \rangle$")
        axs[c].set_xlabel(r"$x / a$")

    def create_inset_axes(ax, region_x, region_y, bounds):
        axins = ax.inset_axes(
            bounds,
            xlim=region_x, ylim=region_y,
            xticks=axs["y"].get_xticks(), xticklabels=[],
            yticks=axs["y"].get_yticks(), yticklabels=[]
        )
        axins.set_xlim(*region_x)
        axins.set_ylim(*region_y)
        for dmi in DMI:
            for d in directions:
                axins.plot(x_space[dmi][d], magn_dict[dmi][d]["y"], **plot_kwargs_dict[dmi],
                           marker="o", markersize=1.5)
        ax.indicate_inset_zoom(axins, edgecolor="black", alpha=1.0, linewidth=1.0)


    create_inset_axes(axs["y"], (-2, 4), (0.00075, 0.00138), [0.04, 0.7, 0.37, 0.27])
    create_inset_axes(axs["y"], (-4, 2), (-0.00133, -0.0007), [0.6, 0.3, 0.37, 0.27])

    legend1 = axs["y"].legend(handles=lines["y"][False], ncols=2, title="no DMI", fontsize="small",
                              loc="lower center")
    axs["y"].add_artist(legend1)

    legend2 = axs["z"].legend(handles=lines["z"][True], ncols=2, title="DMI", fontsize="small",
                              loc="lower center")
    axs["z"].add_artist(legend2)

    fig.tight_layout()

    save_path = f"{save_base_path}{save_name}"
    if save_name:
        fig.savefig(save_path)

    plt.show()

    return save_path



def sse_magnon_accumulation_dmi(accumulation=True):

    paths_dmi = {
        "100": "/data/scc/marian.gunsch/16/AM_xTstep_DMI_T2/",
        "010": "/data/scc/marian.gunsch/16/AM_yTstep_DMI_T2/",
        "110": "/data/scc/marian.gunsch/04/04_AM_tilted_xTstep_DMI_T2-2/",
        "-110": "/data/scc/marian.gunsch/04/04_AM_tilted_yTstep_DMI_T2-2/"
    }

    paths_nodmi = {
        "100": "/data/scc/marian.gunsch/16/AM_xTstep_T2/",
        "010": "/data/scc/marian.gunsch/16/AM_yTstep_T2/",
        "110": "/data/scc/marian.gunsch/04/04_AM_tilted_xTstep_T2-2/",
        "-110": "/data/scc/marian.gunsch/04/04_AM_tilted_yTstep_T2-2/"
    }

    paths_equi_w = {
        False: "/data/scc/marian.gunsch/10/AM_tilt_Tstairs_T2_x-2/",
        True: "/data/scc/marian.gunsch/00/AM_tiltedX_ttmstairs_DMI_ferri-2/"
    }

    path_equi_c_dmi = "/data/scc/marian.gunsch/03/03_AM_tilted_Tstairs_DMI_T0/"

    directions = paths_nodmi.keys()
    DMI = [False, True]
    components = ["y", "z"]

    data_equi = mag_util.npy_files_from_dict(paths_equi_w)

    data_equi_c_dmi = mag_util.npy_files(path_equi_c_dmi)

    data_dmi = mag_util.npy_files_from_dict(paths_dmi)
    data_nodmi = mag_util.npy_files_from_dict(paths_nodmi)

    data = { True : data_dmi, False : data_nodmi }

    magn_equi = dict()
    for dmi in DMI:
        magn_equi[dmi] = dict()
        for c in components:
            S_A = np.mean(mag_util.get_component(data_equi[0][dmi], c))
            S_B = np.mean(mag_util.get_component(data_equi[1][dmi], c))
            magn_equi[dmi][c] = physics.magnetization(S_A, S_B)

    magn_equi_c_dmi = dict()
    for c in components:
        S_A = np.mean(mag_util.get_component(data_equi_c_dmi[0], c))
        S_B = np.mean(mag_util.get_component(data_equi_c_dmi[1], c))
        magn_equi_c_dmi[c] = physics.magnetization(S_A, S_B)

    magn = dict()
    x_space = dict()
    for dmi in DMI:
        magn[dmi] = dict()
        x_space[dmi] = dict()
        for d in directions:
            magn[dmi][d] = dict()
            for c in components:
                S_A = mag_util.get_component(data[dmi][0][d], c)
                S_B = mag_util.get_component(data[dmi][1][d], c)
                magn[dmi][d][c] = physics.magnetization(S_A, S_B, True)

                if not accumulation:
                    continue

                N = len(magn[dmi][d][c])
                i = helper.get_absolute_T_step_index(0.49, N)
                magn[dmi][d][c][:i] -= magn_equi[dmi][c]

                if dmi:     # ground state for dmi not so easy
                    magn[dmi][d][c][i:] -= magn_equi_c_dmi[c]


            N = len(magn[dmi][d]["z"])
            T_step_pos = helper.get_actual_Tstep_pos(0.49, N) - 1
            warnings.warn("T step position still unclear")
            x_space[dmi][d] = np.arange(N) - T_step_pos
            x_space[dmi][d] *= physics.get_lattice_constant(d)


    save_path = plot_sse(x_space, magn, save_name="dmi_sse.pdf")
    save_util.source_paths(save_path, f"SSE: \n\tDMI: {paths_dmi} \n\tno DMI: {paths_nodmi}\n"
                                      f"equilibrium: \n\tDMI: {paths_equi_w[True]} \n\tno DMI: {paths_equi_w[False]}\n"
                                      f"equilibrium cold: \n\tDMI: {path_equi_c_dmi}")



# %% SPIN SEEBECK EFFECT DMI + MAGNETIC FIELD

def sse_magnon_accumulation_dmi_B():
    # TODO!
    paths_dmi_B = {
        "100": "/data/scc/marian.gunsch/16/AM_xTstep_DMI_B_T2/",
        "010": "/data/scc/marian.gunsch/16/AM_yTstep_DMI_B_T2/",
        "110": "/data/scc/marian.gunsch/05/05_AM_tilted_xTstep_DMI_T2_staticB/",
        "-110": "/data/scc/marian.gunsch/05/05_AM_tilted_yTstep_DMI_T2_staticB/"
    }

    paths_dmi = {
        "100": "/data/scc/marian.gunsch/16/AM_xTstep_DMI_T2/",
        "010": "/data/scc/marian.gunsch/16/AM_yTstep_DMI_T2/",
        "110": "/data/scc/marian.gunsch/04/04_AM_tilted_xTstep_DMI_T2-2/",
        "-110": "/data/scc/marian.gunsch/04/04_AM_tilted_yTstep_DMI_T2-2/"
    }

    paths = {
        "100": "/data/scc/marian.gunsch/16/AM_xTstep_T2/",
        "010": "/data/scc/marian.gunsch/16/AM_yTstep_T2/",
        "110": "/data/scc/marian.gunsch/04/04_AM_tilted_xTstep_T2-2/",
        "-110": "/data/scc/marian.gunsch/04/04_AM_tilted_yTstep_T2-2/"
    }

    paths_B = {
        "100": "/data/scc/marian.gunsch/13/13_AM_xTstep_T2_B100/",
        "010": "/data/scc/marian.gunsch/13/13_AM_yTstep_T2_B100/",  # simulation not started
        "110": "/data/scc/marian.gunsch/05/05_AM_tilted_xTstep_T2_B100/",
        "-110": "/data/scc/marian.gunsch/05/05_AM_tilted_yTstep_T2_B100/"
    }
    warnings.warn("Not all simulations are finished!")

    DMI = [False, True]
    magnetic_fields = [False, True]
    directions = paths.keys()

    # paths[B][dmi][direction]
    paths = {
        False : {
            False : paths,
            True : paths_dmi,
        },
        True : {
            False : paths_B,
            True : paths_dmi_B
        }
    }


    mag_util.npy_files_from_dict(paths_dmi_B)


# %% Main

def main():
    pass

    # average_spin_components()

    # dispersion_relation_dmi()

    sse_magnon_accumulation_dmi()

    # sse_magnon_accumulation_dmi_B()