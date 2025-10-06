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
import itertools

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

angle = {
    2: 0.04865704265,
    5: 0.11971502082
}


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


def fancy_equilibrium_comparison_plot(S, magn, titles, dmi_strength):
    arrow_kwargs = dict(arrowstyle="-|>", mutation_scale=10, shrinkA=0, shrinkB=0, lw=1.0)

    DMI = [False, True, "DMI+B", "B"]   # I prefer this order

    fig, axs = plt.subplots(nrows=1, ncols=len(DMI), sharex=True, sharey=True,
                            figsize=(mpl_config.get_width(1.0), mpl_config.get_height(0.8)))
    axs = axs.flatten()

    for i, dmi in enumerate(DMI):
        axs[i].axhline(-1.0, alpha=0.8, color="grey", linestyle="--", linewidth=0.5)
        axs[i].axhline(0.0, alpha=0.8, color="grey", linestyle="--", linewidth=0.5, zorder=0)
        axs[i].axhline(1.0, alpha=0.8, color="grey", linestyle="--", linewidth=0.5)

        start = (0, 0)

        endA = (S[dmi]["A"]["y"], S[dmi]["A"]["z"])
        endB = (S[dmi]["B"]["y"], S[dmi]["B"]["z"])
        endNet = (magn[dmi]["y"], magn[dmi]["z"])

        arrowA = FancyArrowPatch(start, endA, **arrow_kwargs, color="blue")
        arrowB = FancyArrowPatch(start, endB, **arrow_kwargs, color="red")
        arrow_net = FancyArrowPatch(start, endNet, **arrow_kwargs, color="black")

        point_style = dict(marker="o", markeredgecolor="white", markersize=2.0, linestyle="", linewidth=0.03)

        axs[i].add_patch(arrowA)
        axs[i].plot(*endA, color="blue", **point_style)
        axs[i].add_patch(arrowB)
        axs[i].plot(*endB, color="red", **point_style)
        if dmi:
            axs[i].add_patch(arrow_net)
            axs[i].plot(*endNet, color="k", **point_style)

        # --- Add text labels with slight offsets ---
        label_ = lambda text : r"$\langle \vec{S}_{\mathrm{" + text + r"}} \rangle$"
        shiftB = 0.1 if dmi == "DMI+B" else 0.04
        shiftnet = -0.1 if dmi == "B" else -0.06
        axs[i].text(endA[0] - 0.04, 0.8 * endA[1] + 0.06, label_("A"), color="blue", va="center", ha="right")
        axs[i].text(endB[0] + shiftB, 0.8 * endB[1] - 0.06, label_("B"), color="red", va="center", ha="left")
        if dmi:
            axs[i].text(endNet[0] + shiftnet, endNet[1] + 0.06, label_("net"), color="black", va="bottom", ha="center")

        axs[i].set_title(titles[dmi], fontsize="medium")

        # circle = Circle(start, radius=1, fill=False, alpha=0.8, color="grey", linestyle="--")
        # axs[i].add_patch(circle)

    if dmi_strength == 5:
        axs[0].set_xlim(-0.34, 0.23)
    elif dmi_strength == 2:
        axs[0].set_xlim(-0.24, 0.17)
    axs[0].set_ylim(-1.1, 1.1)

    axs[0].set_xlabel(r"$\langle S^y \rangle$")
    axs[1].set_xlabel(r"$\langle S^y \rangle$")
    axs[2].set_xlabel(r"$\langle S^y \rangle$")
    axs[3].set_xlabel(r"$\langle S^y \rangle$")
    axs[0].set_ylabel(r"$\langle S^z \rangle$")

    fig.tight_layout()

    save_path = f"{save_base_path}equi_dmi{dmi_strength}_components_fancy.pdf"
    fig.savefig(save_path)

    plt.show()

    return save_path


def average_spin_components(dmi_strength=2):
    if dmi_strength == 2:
        paths = {
            False: "/data/scc/marian.gunsch/10/AM_tilt_Tstairs_T2_x-2/",
            True: "/data/scc/marian.gunsch/20/AM_Tstairs_DMI2_T2_x/",
            "DMI+B": "/data/scc/marian.gunsch/20/AM_Tstairs_x_B100_DMI_T2/",
            "B": "/data/scc/marian.gunsch/10/AM_Tstairs_T2_x_B100-2/"
        }
    elif dmi_strength == 5:
        paths = {
            False: "/data/scc/marian.gunsch/10/AM_tilt_Tstairs_T2_x-2/",
            True: "/data/scc/marian.gunsch/15/AM_DMI_Tstairs_T2_x/",
            "DMI+B": "/data/scc/marian.gunsch/19/AM_Tstairs_x_B100_DMI_T2/",
            "B": "/data/scc/marian.gunsch/10/AM_Tstairs_T2_x_B100-2/"
        }
    else:
        raise ValueError()

    # labels = {
    #     False: "$D = 0$, $B = 0$",
    #     True: "$D > 0$, $B = 0$",
    #     "DMI+B": "$D > 0$, $B > 0$",
    #     "B": "$D=0$, $B > 0$"
    # }
    labels = {
        False: "$D = 0$, $B = 0$",
        True: rf"$D = \SI{{{dmi_strength}}}{{\meV}}$, $B = 0$",
        "DMI+B": rf"$D = \SI{{{dmi_strength}}}{{\meV}}$, $B > 0$",
        "B": "$D=0$, $B > 0$"
    }

    dataA, dataB = mag_util.npy_files_from_dict(paths)
    data = dict(A=dataA, B=dataB)

    components = ["x", "y", "z"]
    sublattices = ["A", "B"]
    DMI = [False, True, "DMI+B", "B"]

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

    angle = np.abs(np.arctan2(S[True]["A"]["y"], S[True]["A"]["z"]))
    print()
    print(f"{angle=}")
    print()

    # S[dmi][sl][component]
    # magn[dmi][component]

    # save_path = equilibrium_comparison_plot(S, magn, labels)
    # save_util.source_paths(save_path, paths)

    save_path = fancy_equilibrium_comparison_plot(S, magn, labels, dmi_strength)
    save_util.source_paths(save_path, paths)

    # Think about what exactly to compare? Dmi/no DMI for different temperatures? Or just one temperature?


def dispersion_relation_dmi(shading='gouraud', dmi=2, use_angle=False):
    # TODO!
    if dmi == 2:
        paths = {
            "100": "/data/scc/marian.gunsch/20/AM_Tstairs_DMI2_T2_x/",     # not sure whether to use
            "010": "/data/scc/marian.gunsch/20/AM_Tstairs_DMI2_T2_y/",     # not sure whether to use
            "1-10": "/data/scc/marian.gunsch/20/AM_tilt_Tstairs_DMI2_T2_x/",
            "110": "/data/scc/marian.gunsch/20/AM_tilt_Tstairs_DMI2_T2_y/"      # TODO
        }
    elif dmi == 5:
        paths = {
            "100": "/data/scc/marian.gunsch/15/AM_DMI_Tstairs_T2_x/",  # not sure whether to use
            "010": "/data/scc/marian.gunsch/15/AM_DMI_Tstairs_T2_y/",  # not sure whether to use
            "1-10": "/data/scc/marian.gunsch/15/AM_tilt_DMI_Tstairs_T2_x/",
            "110": "/data/scc/marian.gunsch/15/AM_tilt_DMI_Tstairs_T2_y/"
        }
    else:
        raise ValueError()

    paths_nodmi = {
        "100": f"/data/scc/marian.gunsch/10/AM_Tstairs_T2_x-2/",      # not sure whether to use
        "010": f"/data/scc/marian.gunsch/10/AM_Tstairs_T2_y-2/",      # not sure whether to use
        "1-10": f"/data/scc/marian.gunsch/10/AM_tilt_Tstairs_T2_x-2/",
        "110": f"/data/scc/marian.gunsch/10/AM__tilt_Tstairs_T2_y-2/"  # oups: '__'
    }

    directions = paths.keys()
    if directions != paths.keys():
        raise AttributeError("Conflicting keys!")

    from thesis.subsections.equilibrium import dispersion_comparison_table_data, dispersion_comparison_table_plot, band_gap_plot

    if use_angle:
        k_dict, freq_dict, magnon_density_dict, omega1, omega2, band_gap = dispersion_comparison_table_data(paths_nodmi, paths, tilt_angle=angle[dmi])
        band_gap_plot(band_gap)
        dispersion_comparison_table_plot(k_dict, freq_dict, magnon_density_dict,
                                         version=2,
                                         # left_title="no DMI", right_title="DMI",
                                         left_title=r"$D = 0$", right_title=rf"$D = \SI{{{dmi}}}{{\meV}}$",
                                         save_path=f"{save_base_path}dispersion_comparison_dmi{dmi}_table_angl.pdf",
                                         shading=shading)

    k_dict, freq_dict, magnon_density_dict, omega1, omega2, band_gap = dispersion_comparison_table_data(paths_nodmi,
                                                                                                        paths)
    band_gap_plot(band_gap)
    dispersion_comparison_table_plot(k_dict, freq_dict, magnon_density_dict,
                                     version=2,
                                     # left_title="no DMI", right_title="DMI",
                                     left_title=r"$D = 0$", right_title=rf"$D = \SI{{{dmi}}}{{\meV}}$",
                                     save_path=f"{save_base_path}dispersion_comparison_dmi{dmi}_table_noangl.pdf",
                                     shading=shading)



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
    T_step_markings = dict((c, []) for c in components)

    for c in components:
        for dmi in DMI:
            for d in directions:
                _line, = axs[c].plot(x_space[dmi][d], magn_dict[dmi][d][c], label=rf"\hkl[{d}]",
                                       **plot_kwargs_dict[dmi])
                lines[c][dmi].append(_line)

        Tm = plot_util.place_Tstep_marking(axs[c], 0.0)
        T_step_markings[c].append(Tm)


    for c in components:
        axs[c].set_xlim(*xlim)
        axs[c].set_ylim(*ylim)
        axs[c].set_ylabel(rf"$\langle \Delta S^{c} \rangle$")
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
                plot_util.place_Tstep_marking(axins, 0.0)

        # ax.indicate_inset_zoom(axins, edgecolor="black", alpha=1.0, linewidth=1.0)


    create_inset_axes(axs["y"], (-2, 4), (0.00075, 0.00138), [0.04, 0.7, 0.37, 0.27])
    create_inset_axes(axs["y"], (-4, 2), (-0.00133, -0.0007), [0.6, 0.3, 0.37, 0.27])

    legend1 = axs["y"].legend(handles=lines["y"][False], ncols=2, title="no DMI", fontsize="small",
                              loc="lower center")
    axs["y"].add_artist(legend1)

    legend2 = axs["z"].legend(handles=lines["z"][True], ncols=2, title="DMI", fontsize="small",
                              loc="lower center")
    axs["z"].add_artist(legend2)

    T_legend = axs['z'].legend(handles=T_step_markings['z'], loc="upper left")
    axs['z'].add_artist(T_legend)

    fig.tight_layout()

    save_path = f"{save_base_path}{save_name}"
    if save_name:
        fig.savefig(save_path)

    plt.show()

    return save_path



def sse_magnon_accumulation_dmi(accumulation=True, dmi_strength=2):

    raise NotImplementedError("Inset axes wrong. Do not need this function anymore.")

    if dmi_strength == 2:
        paths_dmi = {
            "100": "/data/scc/marian.gunsch/20/AM_xTstep_DMI2_T2/",
            "010": "/data/scc/marian.gunsch/20/AM_yTstep_DMI2_T2/",
            "1-10": "/data/scc/marian.gunsch/20/AM_tilt_xTstep_DMI2_T2-2/",
            "110": "/data/scc/marian.gunsch/20/AM_tilt_yTstep_DMI2_T2-2/"
        }
    else:
        raise NotImplementedError()

    paths_nodmi = {
        "100": "/data/scc/marian.gunsch/16/AM_xTstep_T2/",
        "010": "/data/scc/marian.gunsch/16/AM_yTstep_T2/",
        "1-10": "/data/scc/marian.gunsch/04/04_AM_tilted_xTstep_T2-2/",
        "110": "/data/scc/marian.gunsch/04/04_AM_tilted_yTstep_T2-2/"
    }

    if dmi_strength == 2:
        paths_equi_w = {
            False: "/data/scc/marian.gunsch/10/AM_tilt_Tstairs_T2_x-2/",
            True: "/data/scc/marian.gunsch/20/AM_Tstairs_DMI2_T2_x/"
        }

        path_equi_c_dmi = "/data/scc/marian.gunsch/20/AM_tilt_Tstairs_DMI2_T0/"
    else:
        raise NotImplementedError()

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

                N = magn[dmi][d][c].shape[0]
                i = helper.get_absolute_T_step_index(0.49, N)
                magn[dmi][d][c][:i] -= magn_equi[dmi][c]

                if dmi:     # ground state for dmi not so easy
                    magn[dmi][d][c][i:] -= magn_equi_c_dmi[c]


            N = len(magn[dmi][d]["z"])
            T_step_pos = helper.get_actual_Tstep_pos(0.49, N)
            x_space[dmi][d] = np.arange(N) - T_step_pos
            x_space[dmi][d] *= physics.get_lattice_constant(d)


    save_path = plot_sse(x_space, magn, save_name="dmi_sse.pdf")
    save_util.source_paths(save_path, f"SSE: \n\tDMI: {paths_dmi} \n\tno DMI: {paths_nodmi}\n"
                                      f"equilibrium: \n\tDMI: {paths_equi_w[True]} \n\tno DMI: {paths_equi_w[False]}\n"
                                      f"equilibrium cold: \n\tDMI: {path_equi_c_dmi}")



# %% SPIN SEEBECK EFFECT DMI + MAGNETIC FIELD

def sse_plot_magn_dmi_title_names(B, dmi, dmi_strength=None, spacing=r"\quad"):
    if dmi_strength:
        dmi_str = rf"D = \SI{{{dmi_strength}}}{{\meV}}" if dmi else "D = 0"
    else:
        dmi_str = "D > 0" if dmi else "D = 0"

    B_str = "B > 0" if B else "B = 0"
    return fr"${B_str}\mathrm{{,}} {spacing} {dmi_str}$"


def sse_plot_magn_dmi_y(x_space, magn, xlim=(-86, 86), ylim=(-0.0019, 0.0019), dmi_strength=None):
    c = "y"
    dmi = True
    magnetic_fields = [False, True]
    directions = magn[False][dmi].keys()

    plot_kwargs = dict(linestyle="-", linewidth=0.7, marker="o", markersize=0.8)
    warnings.warn("Think about using markers here.")

    title_name = sse_plot_magn_dmi_title_names

    fig, axs_arr = plt.subplots(ncols=2, sharex=True, sharey=True)
    axs_arr = axs_arr.flatten()
    axs = { False: axs_arr[0], True: axs_arr[1] }

    lines = dict((B, []) for B in magnetic_fields)
    T_step_markings = dict((B, []) for B in magnetic_fields)

    for B in magnetic_fields:
        for d in directions:
            line_, = axs[B].plot(x_space[B][dmi][d], magn[B][dmi][d][c], **plot_kwargs,
                           label=fr"\hkl[{d}]")
            lines[B].append(line_)

        axs[B].axhline(0, color="gray", linestyle="-", marker="", linewidth=0.5, zorder=0)

        Tm = plot_util.place_Tstep_marking(axs[B], 0.0)
        T_step_markings[B].append(Tm)

        axs[B].set_title(title_name(B, dmi, dmi_strength))
        axs[B].set_xlabel("$x / a$")

    def create_inset_axes(B, region_x, region_y, bounds):
        ax = axs[B]
        axins = ax.inset_axes(
            bounds,
            xlim=region_x, ylim=region_y,
            xticks=ax.get_xticks(), xticklabels=[],
            yticks=ax.get_yticks(), yticklabels=[]
        )
        plot_kwargs_ = plot_kwargs.copy()
        plot_kwargs_.update(dict(marker="o", markersize=1))
        axins.set_xlim(*sorted(region_x))
        axins.set_ylim(*sorted(region_y))
        for d in directions:
            axins.plot(x_space[B][dmi][d], magn[B][dmi][d][c], **plot_kwargs_)
            plot_util.place_Tstep_marking(axins, 0.0)

        # ax.indicate_inset_zoom(axins, edgecolor="black", alpha=1.0, linewidth=1.0)

    def place_legend(debug=False):
        # needs to be called after layout is set (so after tight_layout() for example)
        x1 = axs_arr[0].get_position().get_points()[1][0]
        x2 = axs_arr[1].get_position().get_points()[0][0]
        y = axs_arr[0].get_position().get_points()[0][1]
        bbox = (0.5 * (x1 + x2), y)

        if debug:
            fig.text(*bbox, "x", ha="center", va="center", color="violet")
        legend1 = fig.legend(handles=lines[False], ncols=2,
                             loc="lower center", bbox_to_anchor=bbox)
        fig.add_artist(legend1)

        T_legend = axs[False].legend(handles=T_step_markings[False], loc="upper left")
        axs[False].add_artist(T_legend)

    axs[False].set_ylabel(rf"$ \langle \Delta S^{c} \rangle$")
    axs[False].set_xlim(*xlim)
    if ylim:
        axs[False].set_ylim(*ylim)

    if dmi_strength == 5:
        create_inset_axes(False, (-4, 1), (-0.0009, -0.0016), [0.03, 0.03, 0.37, 0.33])
        create_inset_axes(False, (-1, 4), (0.0016, 0.0009), [0.6, 0.64, 0.37, 0.33])
        create_inset_axes(True, (-4, 1), (-0.0009, -0.0016), [0.6, 0.03, 0.37, 0.33])
        create_inset_axes(True, (-1, 4), (0.0016, 0.0007), [0.03, 0.64, 0.37, 0.33])
    else:
        create_inset_axes(False, (-4, 1), (-0.0007, -0.00025), [0.03, 0.03, 0.37, 0.33])
        create_inset_axes(False, (-1, 4), (0.00025, 0.0007), [0.6, 0.64, 0.37, 0.33])
        create_inset_axes(True, (-4, 1), (-0.0007, -0.00025), [0.6, 0.03, 0.37, 0.33])
        create_inset_axes(True, (-1, 4), (0.00025, 0.0007), [0.03, 0.64, 0.37, 0.33])

    fig.tight_layout()

    place_legend(debug=False)

    save_path = f"{save_base_path}sse_magn_dmi{dmi_strength}_B_y.pdf"
    fig.savefig(save_path)

    plt.show()

    return save_path



def sse_plot_magn_dmi_z(x_space, magn, xlim=(-86, 86), ylim=(-0.0021, 0.0017), dmi_strength=None):
    c = "z"
    DMI = [False, True]
    magnetic_fields = [False, True]
    directions = magn[magnetic_fields[0]][DMI[0]].keys()

    plot_kwargs = dict(linestyle="-", linewidth=0.7, marker="o", markersize=0.8)
    warnings.warn("Think about using markers here.")

    title_name = sse_plot_magn_dmi_title_names


    fig, axs_arr = plt.subplots(figsize=(mpl_config.get_width(1.0), 1.5 * mpl_config.get_height(1.0)),
                            nrows=2, ncols=2, sharex=True, sharey=True)
    axs_arr = axs_arr.flatten()
    axs = { False: { False: axs_arr[0], True: axs_arr[1]}, True: {False: axs_arr[2], True: axs_arr[3]} }

    lines = dict((B, dict((dmi, []) for dmi in DMI)) for B in magnetic_fields)
    T_step_markings = dict((B, dict((dmi, []) for dmi in DMI)) for B in magnetic_fields)

    for B, dmi in itertools.product(magnetic_fields, DMI):
        for d in directions:
            line_, = axs[B][dmi].plot(x_space[B][dmi][d], magn[B][dmi][d][c], **plot_kwargs,
                             label=fr"\hkl[{d}]")
            lines[B][dmi].append(line_)

        Tm = plot_util.place_Tstep_marking(axs[B][dmi], 0.0)
        T_step_markings[B][dmi].append(Tm)

        axs[B][dmi].set_title(title_name(B, dmi, dmi_strength))

        # axs[B][dmi].legend()

    def place_legend(debug=False):
        # needs to be called after layout is set (so after tight_layout() for example)
        x1 = axs_arr[0].get_position().get_points()[1][0]
        x2 = axs_arr[1].get_position().get_points()[0][0]
        y = axs_arr[0].get_position().get_points()[0][1]
        bbox = (0.5 * (x1 + x2), y)

        if debug:
            fig.text(*bbox, "x", ha="center", va="center", color="violet")
        legend1 = fig.legend(handles=lines[False][False], ncols=2,
                             loc="lower center", bbox_to_anchor=bbox)
        fig.add_artist(legend1)

        T_legend = axs[True][False].legend(handles=T_step_markings[True][False], loc="lower left")
        axs[True][False].add_artist(T_legend)


    axs_arr[0].set_xlim(*xlim)
    if ylim:
        axs_arr[0].set_ylim(*ylim)

    axs_arr[0].set_ylabel(rf"$ \langle \Delta S^{c} \rangle$")
    axs_arr[2].set_ylabel(rf"$ \langle \Delta S^{c} \rangle$")
    axs_arr[2].set_xlabel("$x / a$")
    axs_arr[3].set_xlabel("$x / a$")

    fig.tight_layout()

    place_legend(False)

    plt.show()

    save_path = f"{save_base_path}sse_magn_dmi{dmi_strength}_B_z_.pdf"
    fig.savefig(save_path)

    plt.show()

    return save_path


def sse_magnon_accumulation_dmi_B(accumulation=True, dmi_strength=2):
    if not accumulation:
        raise NotImplementedError("Plots are only properly implemented for accumulation!")

    if dmi_strength == 2:
        paths_dmi_B = {
            "100": "/data/scc/marian.gunsch/20/AM_xTstep_DMI2_B_T2/",
            "010": "/data/scc/marian.gunsch/20/AM_yTstep_DMI2_B_T2/",
            "1-10": "/data/scc/marian.gunsch/20/AM_tilt_xTstep_DMI2_T2_B/",
            "110": "/data/scc/marian.gunsch/20/AM_tilt_yTstep_DMI2_T2_B/"
        }

        paths_dmi = {
            "100": "/data/scc/marian.gunsch/20/AM_xTstep_DMI2_T2/",
            "010": "/data/scc/marian.gunsch/20/AM_yTstep_DMI2_T2/",
            "1-10": "/data/scc/marian.gunsch/20/AM_tilt_xTstep_DMI2_T2-2/",
            "110": "/data/scc/marian.gunsch/20/AM_tilt_yTstep_DMI2_T2-2/"
        }
    elif dmi_strength == 5:
        paths_dmi_B = {
            "100": "/data/scc/marian.gunsch/16/AM_xTstep_DMI_B_T2/",
            "010": "/data/scc/marian.gunsch/16/AM_yTstep_DMI_B_T2/",
            "1-10": "/data/scc/marian.gunsch/05/05_AM_tilted_xTstep_DMI_T2_staticB/",
            "110": "/data/scc/marian.gunsch/05/05_AM_tilted_yTstep_DMI_T2_staticB/"
        }

        paths_dmi = {
            "100": "/data/scc/marian.gunsch/16/AM_xTstep_DMI_T2/",
            "010": "/data/scc/marian.gunsch/16/AM_yTstep_DMI_T2/",
            "1-10": "/data/scc/marian.gunsch/04/04_AM_tilted_xTstep_DMI_T2-2/",
            "110": "/data/scc/marian.gunsch/04/04_AM_tilted_yTstep_DMI_T2-2/"
        }
    else:
        raise ValueError()

    paths = {
        "100": "/data/scc/marian.gunsch/16/AM_xTstep_T2/",
        "010": "/data/scc/marian.gunsch/16/AM_yTstep_T2/",
        "1-10": "/data/scc/marian.gunsch/04/04_AM_tilted_xTstep_T2-2/",
        "110": "/data/scc/marian.gunsch/04/04_AM_tilted_yTstep_T2-2/"
    }

    paths_B = {
        "100": "/data/scc/marian.gunsch/13/13_AM_xTstep_T2_B100/",
        "010": "/data/scc/marian.gunsch/13/13_AM_yTstep_T2_B100/",  # simulation not started
        "1-10": "/data/scc/marian.gunsch/05/05_AM_tilted_xTstep_T2_B100/",
        "110": "/data/scc/marian.gunsch/05/05_AM_tilted_yTstep_T2_B100/"
    }

    DMI = [False, True]
    magnetic_fields = [False, True]
    directions = paths.keys()
    components = ["y", "z"]

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

    paths_equi = {
        False: {
            False: "/data/scc/marian.gunsch/10/AM_tilt_Tstairs_T2_x-2/",
            True: "/data/scc/marian.gunsch/20/AM_Tstairs_DMI2_T2_x/"
        },
        True: {
            False: "/data/scc/marian.gunsch/10/AM_Tstairs_T2_x_B100-2/",
            True: "/data/scc/marian.gunsch/20/AM_Tstairs_x_B100_DMI_T2/"        
        }
    }

    if dmi_strength == 2:
        pass
    elif dmi_strength == 5:
        paths_equi[False][True] = "/data/scc/marian.gunsch/15/AM_DMI_Tstairs_T2_x/"
        paths_equi[True][True] = "/data/scc/marian.gunsch/19/AM_Tstairs_x_B100_DMI_T2/"
    else:
        raise ValueError()

    if dmi_strength == 2:
        path_equi_dmi_cold = {
            False: "/data/scc/marian.gunsch/20/AM_tilt_Tstairs_DMI2_T0/",
            True: "/data/scc/marian.gunsch/20/AM_Tstairs_x_B100_DMI_T0/"
        }
    elif dmi_strength == 5:
        path_equi_dmi_cold = {
            False: "/data/scc/marian.gunsch/03/03_AM_tilted_Tstairs_DMI_T0/",
            True: "/data/scc/marian.gunsch/19/AM_Tstairs_x_B100_DMI_T0/"
        }
    else:
        raise ValueError()

    data = dict()
    for B in magnetic_fields:
        data[B] = dict()
        for dmi in DMI:
            dataA, dataB = mag_util.npy_files_from_dict(paths[B][dmi])
            data[B][dmi] = dict(A=dataA, B=dataB)

    real_space = dict()
    magn = dict()
    for B in magnetic_fields:   # nested for loops ftw
        magn[B] = dict()
        real_space[B] = dict()
        for dmi in DMI:
            magn[B][dmi] = dict()
            real_space[B][dmi] = dict()
            for d in directions:
                magn[B][dmi][d] = dict()
                for c in components:
                    S_A = mag_util.get_component(data[B][dmi]["A"][d], c)
                    S_B = mag_util.get_component(data[B][dmi]["B"][d], c)
                    magn[B][dmi][d][c] = physics.magnetization(S_A, S_B, True)

                N = magn[B][dmi][d]["z"].shape[0]
                T_step_pos = helper.get_actual_Tstep_pos(0.49, N)
                real_space[B][dmi][d] = np.arange(N) - T_step_pos
                real_space[B][dmi][d] *= physics.get_lattice_constant(d)


    if accumulation:
        magn_equi = dict()
        for B in magnetic_fields:
            magn_equi[B] = dict()
            for dmi in DMI:
                magn_equi[B][dmi] = dict()
                dataA, dataB = mag_util.npy_files(paths_equi[B][dmi])
                for c in components:
                    S_A = np.mean(mag_util.get_component(dataA, c))
                    S_B = np.mean(mag_util.get_component(dataB, c))
                    magn_equi[B][dmi][c] = physics.magnetization(S_A, S_B)


        data_equi_dmi_cold = mag_util.npy_files_from_dict(path_equi_dmi_cold)
        magn_equi_dmi_cold = dict()
        for B in magnetic_fields:
            magn_equi_dmi_cold[B] = dict()
            for c in components:
                S_A = np.mean(mag_util.get_component(data_equi_dmi_cold[0][B], c))
                S_B = np.mean(mag_util.get_component(data_equi_dmi_cold[1][B], c))
                magn_equi_dmi_cold[B][c] = physics.magnetization(S_A, S_B)


        for B, dmi, d, c in itertools.product(magnetic_fields, DMI, directions, components):
            N = magn[B][dmi][d][c].shape[0]
            Tstep = helper.get_absolute_T_step_index(0.49, N)
            magn[B][dmi][d][c][:Tstep] -= magn_equi[B][dmi][c]
            if dmi:
                magn[B][dmi][d][c][Tstep:] -= magn_equi_dmi_cold[B][c]


    paths_str = f"{paths=}, {paths_equi=}, {path_equi_dmi_cold=}"

    save_path = sse_plot_magn_dmi_y(real_space, magn, dmi_strength=dmi_strength)
    save_util.source_paths(save_path, paths_str)

    save_path = sse_plot_magn_dmi_z(real_space, magn, dmi_strength=dmi_strength)
    save_util.source_paths(save_path, paths_str)


# %%

def sse_plot_magn_dmi_z_2(x_space, magn, xlim=(-76, 76), ylim=(-0.0021, 0.0017)):
    c = "z"
    DMI = [0, 2, 5]
    magnetic_fields = [False, True]
    directions = magn[magnetic_fields[0]][DMI[0]].keys()

    plot_kwargs = dict(linestyle="-", linewidth=0.7, marker="o", markersize=0.8)
    warnings.warn("Think about using markers here.")

    title_name = sse_plot_magn_dmi_title_names

    fig, axs_arr = plt.subplots(figsize=(mpl_config.get_width(1.0), 1.5 * mpl_config.get_height(1.0)),
                                nrows=2, ncols=3, sharex=True, sharey=True)
    axs_arr = axs_arr.flatten()
    axs = {False: {0: axs_arr[0], 2: axs_arr[1], 5: axs_arr[2]}, True: {0: axs_arr[3], 2: axs_arr[4], 5: axs_arr[5]}}

    lines = dict((B, dict((dmi, []) for dmi in DMI)) for B in magnetic_fields)
    T_step_markings = dict((B, dict((dmi, []) for dmi in DMI)) for B in magnetic_fields)

    for B, dmi in itertools.product(magnetic_fields, DMI):
        for d in directions:
            line_, = axs[B][dmi].plot(x_space[B][dmi][d], magn[B][dmi][d][c], **plot_kwargs,
                                      label=fr"\hkl[{d}]")
            lines[B][dmi].append(line_)

        axs[B][dmi].axhline(0, color="gray", linestyle="-", marker="", linewidth=0.5, zorder=0)

        Tm = plot_util.place_Tstep_marking(axs[B][dmi], 0.0)
        T_step_markings[B][dmi].append(Tm)

        axs[B][dmi].set_title(title_name(B, dmi, dmi, r"\:"))

        # axs[B][dmi].legend()

    def place_legend(debug=False):
        # needs to be called after layout is set (so after tight_layout() for example)
        x1 = axs_arr[1].get_position().get_points()[0][0]
        x2 = axs_arr[1].get_position().get_points()[1][0]
        y = axs_arr[1].get_position().get_points()[0][1]
        bbox = (0.5 * (x1 + x2), y)

        if debug:
            fig.text(*bbox, "x", ha="center", va="center", color="violet")
        legend1 = fig.legend(handles=lines[False][0], ncols=2,
                             loc="lower center", bbox_to_anchor=bbox)
        fig.add_artist(legend1)

        T_legend = axs[True][0].legend(handles=T_step_markings[True][0], loc="lower left")
        axs[True][0].add_artist(T_legend)

    axs_arr[0].set_xlim(*xlim)
    if ylim:
        axs_arr[0].set_ylim(*ylim)

    axs_arr[0].set_ylabel(rf"$ \langle \Delta S^{c} \rangle$")
    axs_arr[3].set_ylabel(rf"$ \langle \Delta S^{c} \rangle$")
    axs_arr[3].set_xlabel("$x / a$")
    axs_arr[4].set_xlabel("$x / a$")
    axs_arr[5].set_xlabel("$x / a$")

    fig.tight_layout()

    place_legend(False)

    plt.show()

    save_path = f"{save_base_path}sse_magn_dmi_B_z.pdf"
    fig.savefig(save_path)

    plt.show()

    return save_path




def sse_magnon_accumulation_dmi_B_2():
    paths_dmi2_B = {
        "100": "/data/scc/marian.gunsch/20/AM_xTstep_DMI2_B_T2/",
        "010": "/data/scc/marian.gunsch/20/AM_yTstep_DMI2_B_T2/",
        "1-10": "/data/scc/marian.gunsch/20/AM_tilt_xTstep_DMI2_T2_B/",
        "110": "/data/scc/marian.gunsch/20/AM_tilt_yTstep_DMI2_T2_B/"
    }

    paths_dmi2 = {
        "100": "/data/scc/marian.gunsch/20/AM_xTstep_DMI2_T2/",
        "010": "/data/scc/marian.gunsch/20/AM_yTstep_DMI2_T2/",
        "1-10": "/data/scc/marian.gunsch/20/AM_tilt_xTstep_DMI2_T2-2/",
        "110": "/data/scc/marian.gunsch/20/AM_tilt_yTstep_DMI2_T2-2/"
    }

    paths_dmi5_B = {
        "100": "/data/scc/marian.gunsch/16/AM_xTstep_DMI_B_T2/",
        "010": "/data/scc/marian.gunsch/16/AM_yTstep_DMI_B_T2/",
        "1-10": "/data/scc/marian.gunsch/05/05_AM_tilted_xTstep_DMI_T2_staticB/",
        "110": "/data/scc/marian.gunsch/05/05_AM_tilted_yTstep_DMI_T2_staticB/"
    }

    paths_dmi5 = {
        "100": "/data/scc/marian.gunsch/16/AM_xTstep_DMI_T2/",
        "010": "/data/scc/marian.gunsch/16/AM_yTstep_DMI_T2/",
        "1-10": "/data/scc/marian.gunsch/04/04_AM_tilted_xTstep_DMI_T2-2/",
        "110": "/data/scc/marian.gunsch/04/04_AM_tilted_yTstep_DMI_T2-2/"
    }

    paths = {
        "100": "/data/scc/marian.gunsch/16/AM_xTstep_T2/",
        "010": "/data/scc/marian.gunsch/16/AM_yTstep_T2/",
        "1-10": "/data/scc/marian.gunsch/04/04_AM_tilted_xTstep_T2-2/",
        "110": "/data/scc/marian.gunsch/04/04_AM_tilted_yTstep_T2-2/"
    }

    paths_B = {
        "100": "/data/scc/marian.gunsch/13/13_AM_xTstep_T2_B100/",
        "010": "/data/scc/marian.gunsch/13/13_AM_yTstep_T2_B100/",  # simulation not started
        "1-10": "/data/scc/marian.gunsch/05/05_AM_tilted_xTstep_T2_B100/",
        "110": "/data/scc/marian.gunsch/05/05_AM_tilted_yTstep_T2_B100/"
    }

    DMI = [0, 2, 5]
    magnetic_fields = [False, True]
    directions = paths.keys()
    components = ["y", "z"]

    # paths[B][dmi][direction]
    paths = {
        False : {
            0 : paths,
            2 : paths_dmi2,
            5 : paths_dmi5,
        },
        True : {
            0 : paths_B,
            2 : paths_dmi2_B,
            5 : paths_dmi5_B,
        }
    }

    paths_equi = {
        False: {
            0: "/data/scc/marian.gunsch/10/AM_tilt_Tstairs_T2_x-2/",
            2: "/data/scc/marian.gunsch/20/AM_Tstairs_DMI2_T2_x/",
            5: "/data/scc/marian.gunsch/15/AM_DMI_Tstairs_T2_x/"
        },
        True: {
            0: "/data/scc/marian.gunsch/10/AM_Tstairs_T2_x_B100-2/",
            2: "/data/scc/marian.gunsch/20/AM_Tstairs_x_B100_DMI_T2/",
            5: "/data/scc/marian.gunsch/19/AM_Tstairs_x_B100_DMI_T2/"
        }
    }

    paths_equi_dmi_cold = {
        False: {
            2: "/data/scc/marian.gunsch/20/AM_tilt_Tstairs_DMI2_T0/",
            5: "/data/scc/marian.gunsch/03/03_AM_tilted_Tstairs_DMI_T0/"
        },
        True: {
            2: "/data/scc/marian.gunsch/20/AM_Tstairs_x_B100_DMI_T0/",
            5: "/data/scc/marian.gunsch/19/AM_Tstairs_x_B100_DMI_T0/"
        }
    }


    data = dict()
    for B in magnetic_fields:
        data[B] = dict()
        for dmi in DMI:
            dataA, dataB = mag_util.npy_files_from_dict(paths[B][dmi])
            data[B][dmi] = dict(A=dataA, B=dataB)

    real_space = dict()
    magn = dict()
    for B in magnetic_fields:   # nested for loops ftw
        magn[B] = dict()
        real_space[B] = dict()
        for dmi in DMI:
            magn[B][dmi] = dict()
            real_space[B][dmi] = dict()
            for d in directions:
                magn[B][dmi][d] = dict()
                for c in components:
                    S_A = mag_util.get_component(data[B][dmi]["A"][d], c)
                    S_B = mag_util.get_component(data[B][dmi]["B"][d], c)
                    magn[B][dmi][d][c] = physics.magnetization(S_A, S_B, True)

                N = magn[B][dmi][d]["z"].shape[0]
                T_step_pos = helper.get_actual_Tstep_pos(0.49, N)
                real_space[B][dmi][d] = np.arange(N) - T_step_pos
                real_space[B][dmi][d] *= physics.get_lattice_constant(d)


    magn_equi = dict()
    for B in magnetic_fields:
        magn_equi[B] = dict()
        for dmi in DMI:
            magn_equi[B][dmi] = dict()
            dataA, dataB = mag_util.npy_files(paths_equi[B][dmi])
            for c in components:
                S_A = np.mean(mag_util.get_component(dataA, c))
                S_B = np.mean(mag_util.get_component(dataB, c))
                magn_equi[B][dmi][c] = physics.magnetization(S_A, S_B)

    magn_equi_dmi_cold = dict()
    for B in magnetic_fields:
        magn_equi_dmi_cold[B] = dict()
        for dmi in DMI:
            if dmi == 0:
                continue
            magn_equi_dmi_cold[B][dmi] = dict()
            dataA, dataB = mag_util.npy_files(paths_equi_dmi_cold[B][dmi])
            for c in components:
                S_A = np.mean(mag_util.get_component(dataA, c))
                S_B = np.mean(mag_util.get_component(dataB, c))
                magn_equi_dmi_cold[B][dmi][c] = physics.magnetization(S_A, S_B)



    for B, dmi, d, c in itertools.product(magnetic_fields, DMI, directions, components):
        N = magn[B][dmi][d][c].shape[0]
        Tstep = helper.get_absolute_T_step_index(0.49, N)
        magn[B][dmi][d][c][:Tstep] -= magn_equi[B][dmi][c]
        if dmi > 0:
            magn[B][dmi][d][c][Tstep:] -= magn_equi_dmi_cold[B][dmi][c]


    paths_str = f"{paths=}, {paths_equi=}, {paths_equi_dmi_cold=}"

    # save_path = sse_plot_magn_dmi_y(real_space, magn)
    # save_util.source_paths(save_path, paths_str)

    save_path = sse_plot_magn_dmi_z_2(real_space, magn)
    save_util.source_paths(save_path, paths_str)



# %% Main

def main():
    pass

    average_spin_components(dmi_strength=2)
    average_spin_components(dmi_strength=5)

    sse_magnon_accumulation_dmi_B(True, dmi_strength=2)
    sse_magnon_accumulation_dmi_B(True, dmi_strength=5)

    sse_magnon_accumulation_dmi_B_2()

    # dispersion_relation_dmi(dmi=2)
    # dispersion_relation_dmi(dmi=5, use_angle=True)

    # dispersion_relation_dmi(dmi=5, use_angle=False)


