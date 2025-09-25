import warnings

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.constants
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.colors as colors
from matplotlib.collections import PolyCollection
import matplotlib.ticker as ticker

import src.utility as util
import src.mag_util as mag_util
import src.bulk_util as bulk_util
import src.plot_util as plot_util
import src.physics as physics
import src.spinconf_util as spinconf_util
import src.helper as helper
import thesis.mpl_configuration as mpl_conf

# %%

save_base_path = "out/thesis/equilibrium/"

# dispersion_data_points = 1_000
dispersion_data_points = 300_000

if dispersion_data_points < 100_000:
    warnings.warn("Running with limited amount of data points!")


# %% INTRODUCTION OF A STATIC MAGNETIC FIELD
pass
pass


# %% Equilibrium averages

def equilibrium_comparison_Bfield_plot(Sx, Sy, Sz, magn):
    plot_kwargs = dict(marker="o", linestyle="-") #, linewidth=1.5)

    fig = plt.figure()
    gs = fig.add_gridspec(nrows=3, height_ratios=(2, 1.1, 1.1), hspace=0.1,
                          left=0.15, right=0.9, bottom=0.15, top=0.9)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

    fields = sorted(Sz["A"].keys())

    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    ax1.set_ylabel(r"$\langle S^z_{\mathrm{net}} \rangle$")
    ax1.plot(fields, [magn[B] for B in fields], label=r"$m_{\mathrm{net}}$", **plot_kwargs)

    ax2.set_ylabel(r"$\langle S^z_{\mathrm{A}} \rangle$")
    ax2.plot(fields, [Sz["A"][B] for B in fields], label=r"$m_{\mathrm{A}}$", **plot_kwargs)

    ax3.set_ylabel(r"$- \langle S^z_{\mathrm{B}} \rangle$")
    ax3.plot(fields, [-Sz["B"][B] for B in fields], label=r"$-m_{\mathrm{B}}$", **plot_kwargs)

    ax1.label_outer()
    ax2.label_outer()

    ax3.set_xticks(fields)
    ax3.set_xlabel("$B$ (T)")

    fig.savefig(f"out/thesis/equilibrium/comparison_B_field.pdf")

    plt.show()


def equilibrium_comparison_Bfield():
    print("\n\nEQUILIBRIUM COMPARISON: MAGNETIC FIELD")

    paths = {
        -100: "/data/scc/marian.gunsch/06/06_AM_tilted_Tstairs_T2_x_Bn100/",
        -50: "/data/scc/marian.gunsch/10/AM_Tstairs_T2_Bn50",
        0: "/data/scc/marian.gunsch/10/AM_tilt_Tstairs_T2_x-2/",
        50: "/data/scc/marian.gunsch/10/AM_Tstairs_T2_B50",
        100: "/data/scc/marian.gunsch/10/AM_Tstairs_T2_x_B100-2/"
    }

    field_strengths = paths.keys()

    dataA, dataB = mag_util.npy_files_from_dict(paths)
    data = dict(A=dataA, B=dataB)

    Sx = dict(A=dict(), B=dict())
    Sy = dict(A=dict(), B=dict())
    Sz = dict(A=dict(), B=dict())
    magn = dict()

    for B_field in field_strengths:
        for sl in ["A", "B"]:
            Sx[sl][B_field] = np.mean(mag_util.get_component(data[sl][B_field], "x"))
            Sy[sl][B_field] = np.mean(mag_util.get_component(data[sl][B_field], "y"))
            Sz[sl][B_field] = np.mean(mag_util.get_component(data[sl][B_field], "z"))
        magn[B_field] = physics.magnetization(Sz["A"][B_field], Sz["B"][B_field])

    equilibrium_comparison_Bfield_plot(Sx, Sy, Sz, magn)

    warnings.warn("Think about introducing a linear fit here!")



# %% Dispersion relation comparison for B=100T

def dispersion_comparison_table_plot(k_dict, freq_dict, magnon_density_dict, omega1=None, omega2=None, version=2,
                                     left_title=r"$B = 0$", right_title=r"$B > 0$",
                                     save_path=None, shading="gouraud", vmin_=None, vmax_=None):
    if not version in [1, 2]:
        raise ValueError("version must be 1 or 2")

    print(f"Plotting version {version}...")

    rasterized = True
    j0 = 0 if version == 1 else 1

    fig = plt.figure(figsize=(mpl_conf.get_width(1.0), mpl_conf.get_tex_height(0.83)))
    # fig = plt.figure(figsize=mpl_conf.get_size(1.0, None, False))

    if version == 1:
        gs = fig.add_gridspec(nrows=5, ncols=2, height_ratios=[4, 4, 4, 4, 0.2], hspace=0.65, wspace=0.05)
    else:
        gs = fig.add_gridspec(nrows=6, ncols=3, width_ratios=[1.5, 3, 3], height_ratios=[4, 4, 4, 4, 1.2, 0.2],
                              hspace=0.07, wspace=0.05, left=0.05, bottom=0.07, top=0.95, right=0.96
                              )

    axs = np.empty((4, 2), dtype=object)
    for i in range(4):
        for j in range(2):
            if i == j == 0:
                axs[i, j] = fig.add_subplot(gs[i, j + j0])
            elif version == 1:
                axs[i, j] = fig.add_subplot(gs[i, j + j0], sharey=axs[0, 0])
            else:
                axs[i, j] = fig.add_subplot(gs[i, j + j0], sharey=axs[0, 0], sharex=axs[0, 0])

            axs[i, j].xaxis.set_major_locator(ticker.MultipleLocator(base=np.pi / 2))
            axs[i, j].xaxis.set_major_formatter(ticker.FuncFormatter(plot_util.multiple_of_pi_over_2))

        axs[i, 1].tick_params(labelleft=False)

    if version == 2:
        for i in range(3):
            for j in range(2):
                axs[i, j].tick_params(labelbottom=False)

    # colorbar axis spans across bottom
    cax = fig.add_subplot(gs[-1, j0:])

    fields = k_dict.keys()
    directions = k_dict[False]

    axs_dict = dict(
        (B, dict(
            (direction, axs[i, j])
            for direction, i in zip(directions, range(4))))
        for B, j in zip(fields, range(2))
    )

    axs_dict[False]["100"].set_title(left_title)
    axs_dict[True]["100"].set_title(right_title)

    min_magn_dens = np.inf
    max_magn_dens = - np.inf
    for field in fields:
        for direction in directions:
            magnon_density = magnon_density_dict[field][direction]
            min_magn_dens = min(magnon_density.min(), min_magn_dens)
            max_magn_dens = max(magnon_density.max(), max_magn_dens)

    if vmin_:
        min_magn_dens = vmin_
    if vmax_:
        max_magn_dens = vmax_

    im_list = []

    print("[", end="")
    for field in fields:
        for direction in directions:
            print("-", end="")
            ax = axs_dict[field][direction]
            k_vectors = k_dict[field][direction]
            freqs = freq_dict[field][direction]

            freqs *= 1e-15
            if not field:
                ax.set_ylabel(r"$\omega$ (\SI{e15}{\radian\per\second})")

            magnon_density = magnon_density_dict[field][direction]
            im = ax.pcolormesh(k_vectors, freqs, magnon_density, shading=shading,
                               norm=colors.LogNorm(vmin=min_magn_dens, vmax=max_magn_dens),
                               rasterized=rasterized)
            # print("Using shading='gouraud'")

            im_list.append(im)

            if omega1 and omega2:
                l = rf"$\omega^{{\hkl[{direction}]}}"
                w1 = omega1[field][direction]
                w2 = omega2[field][direction]
                ax.plot(k_vectors, w1, label=fr"{l}_\mathrm{{A}}$", color="blue", linewidth=0.5)
                ax.plot(k_vectors, w2, label=fr"{l}_\mathrm{{A}}$", color="red", linewidth=0.5)
                ax.legend(loc="upper right")

            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")


    def set_xlabels_1():
        for field in fields:
            for direction in directions:
                ax = axs_dict[field][direction]
                a = r"\tilde{a}" if direction in ["1-10", "110"] else r"a"
                ax.set_xlabel(rf"$k \cdot {a}$ in \hkl[{direction}]")

    def set_xlabels_2():
        for field in fields:
            ax = axs_dict[field]["110"]
            ax.set_xlabel(r"$k \cdot a_{\mathrm{d}}$")
        gs_dict = dict((direction, gs[i, 0]) for direction, i in zip(directions, range(4)))
        for direction in directions:
            bbox = gs_dict[direction].get_position(fig)
            x = bbox.x0
            y = (bbox.y0 + bbox.y1) / 2
            fig.text(x, y, rf"\hkl[{direction}]", ha="left", va="center", fontsize=mpl.rcParams["axes.titlesize"],
                     rotation=90)

    if version == 1:
        set_xlabels_1()
    else:
        set_xlabels_2()

    print("-", end="")
    cb = fig.colorbar(im_list[-1], cax=cax, orientation="horizontal")
    cb.set_label(r"magnon density $n(\omega, k)$ (arb. unit)")
    print("]")

    if save_path:
        print("Saving fig...")
        fig.savefig(f"{save_path[:-4]}_{shading[:4]}.pdf")

        # from thesis.theoretical_figures import crop_to_size
        # crop_to_size(save_path)


    # print("Showing fig...")
    # plt.show()


def get_band_gap(omega1, omega2):
    return np.min(np.abs(omega1)) + np.min(np.abs(omega2))


def get_omega_k(f, m):
    h = int(m.shape[0] * 0.5)
    imax1 = np.argmax(m[:h], axis=0)
    imax2 = np.argmax(m[-h:], axis=0)
    omega1 = f[imax1]
    omega2 = f[-h:][imax2]

    return omega1, omega2

def get_spectral_power(m, min_=1e4):
    h = int(m.shape[0] * 0.5)
    return np.sum(m[m>min_][:h]), np.sum(m[m>min_][-h:])

def get_magnon_number_from_rayleighjeans(k, omega_k, TmeV):
    T = TmeV * scipy.constants.eV * scipy.constants.milli
    kB = scipy.constants.k
    hbar = scipy.constants.hbar
    def rayleigh_jeans(omega_k):
        return kB * T / (hbar * omega_k)

    n_k = rayleigh_jeans(omega_k)
    return np.sum(n_k)


def band_gap_plot(band_gap):
    fig, ax = plt.subplots()
    for field in band_gap.keys():
        gaps = band_gap[field]
        ax.plot(list(gaps.keys()), [gaps[d] for d in gaps.keys()], label=f"{field=}")
    ax.legend()
    plt.show()


def temp_ana(k, f, m):
    print(f"{m.shape}")
    h = int(m.shape[0] * 0.5)
    print(f"{np.sum(m[:h])=}")
    print(f"{np.sum(m[m>1e4][:h])=}")
    print(f"{np.sum(m[-h:])=}")
    print(f"{np.sum(m[m>1e4][-h:])=}")

    imax1 = np.argmax(m[:h], axis=0)
    imax2 = np.argmax(m[-h:], axis=0)
    omega1 = f[imax1]
    omega2 = f[-h:][imax2]
    # fig, ax = plt.subplots()
    # ax.plot(k, omega1)
    # ax.plot(k, omega2)
    # plt.show()

    Sk1 = m[imax1]
    Sk2 = m[-h:][imax2]

    nk1 = Sk1 / omega1
    print(np.sum(nk1))
    nk2 = Sk2 / omega2
    print(np.sum(nk2))




def dispersion_comparison_table_data(paths_no, paths_yes):

    delta_x = {
        "100": physics.lattice_constant,
        "010": physics.lattice_constant,
        "1-10": physics.lattice_constant,
        "110": physics.lattice_constant
    }

    directions = paths_yes.keys()

    dataA_no, dataB_no = mag_util.npy_files_from_dict(paths_no, slice_index=-dispersion_data_points,
                                                  max_rows=dispersion_data_points + 10000)
    dataA_yes, dataB_yes = mag_util.npy_files_from_dict(paths_yes, slice_index=-dispersion_data_points,
                                                  max_rows=dispersion_data_points + 10000)

    print("Finished loading data. Processing...")

    data_dict = {
        False: dict(A=dataA_no, B=dataB_no),  # No magnetic field
        True: dict(A=dataA_yes, B=dataB_yes)  # Yes magnetic field
    }

    paths = {
        False: paths_no,
        True: paths_yes
    }

    k_dict = {
        False: dict(),
        True: dict()
    }
    freq_dict = {
        False: dict(),
        True: dict()
    }
    magnon_density_dict = {
        False: dict(),
        True: dict()
    }

    omega1 = { False: dict(), True: dict()}
    omega2 = { False: dict(), True: dict()}
    band_gap = { False: dict(), True: dict()}

    for magnetic_field in data_dict.keys():
        for direction in directions:
            print(f"{magnetic_field=}, {direction=}")
            data_A = data_dict[magnetic_field]["A"][direction]
            data_B = data_dict[magnetic_field]["B"][direction]

            data_points = min(data_A.shape[0], data_B.shape[0], dispersion_data_points)

            if data_points < dispersion_data_points:
                warnings.warn(f"{magnetic_field}, [{direction}]: Can only run with {data_points} data points.")

            Sx = physics.magnetization(mag_util.get_component(data_A[:data_points], "x"),
                                       mag_util.get_component(data_B[:data_points], "x"))
            Sy = physics.magnetization(mag_util.get_component(data_A[:data_points], "y"),
                                       mag_util.get_component(data_B[:data_points], "y"))
            dx = delta_x[direction]
            dt = util.get_time_step(paths[magnetic_field][direction])
            k, f, m = physics.dispersion(Sx, Sy, dx, dt)

            temp_ana(k, f, m)

            omega1[magnetic_field][direction], omega2[magnetic_field][direction] = get_omega_k(f, m)
            band_gap[magnetic_field][direction] = get_band_gap(omega1[magnetic_field][direction],
                                                               omega2[magnetic_field][direction])
            print(f"{get_magnon_number_from_rayleighjeans(k, omega1[magnetic_field][direction], 2)=}")
            print(f"{get_magnon_number_from_rayleighjeans(k, omega2[magnetic_field][direction], 2)=}")
            print(f"{get_spectral_power(m, 1e4)=}")
            print(f"{get_spectral_power(m, 0)=}")

            k_dict[magnetic_field][direction] = k
            freq_dict[magnetic_field][direction] = f
            magnon_density_dict[magnetic_field][direction] = m

    return k_dict, freq_dict, magnon_density_dict, omega1, omega2, band_gap



def dispersion_comparison_Bfield_table_data():
    paths_noB = {
        "100": f"/data/scc/marian.gunsch/10/AM_Tstairs_T2_x-2/",
        "010": f"/data/scc/marian.gunsch/10/AM_Tstairs_T2_y-2/",
        "1-10": f"/data/scc/marian.gunsch/10/AM_tilt_Tstairs_T2_x-2/",
        "110": f"/data/scc/marian.gunsch/10/AM__tilt_Tstairs_T2_y-2/"  # oups
    }
    paths_B = {
        "100": f"/data/scc/marian.gunsch/10/AM_Tstairs_T2_x_B100-2/",
        "010": f"/data/scc/marian.gunsch/10/AM_Tstairs_T2_y_B100-2/",
        "1-10": f"/data/scc/marian.gunsch/10/AM_tilt_Tstairs_T2_x_B100-2/",
        "110": f"/data/scc/marian.gunsch/10/AM_tilt_Tstairs_T2_y_B100-2/"
    }
    return dispersion_comparison_table_data(paths_noB, paths_B)


def dispersion_comparison_Bfield_table(version=1, shading='gouraud'):
    print("\n\nDISPERSION RELATION COMPARISON: MAGNETIC FIELD")

    k_dict, freq_dict, magnon_density_dict, omega1, omega2, band_gap = dispersion_comparison_Bfield_table_data()
    dispersion_comparison_table_plot(k_dict, freq_dict, magnon_density_dict, omega1, omega2, version=version,
                                     save_path=f"{save_base_path}dispersion_comparison_Bfield_table.pdf",
                                     shading=shading, vmin_=1e-3)
    band_gap_plot(band_gap)

# %% Comparison of dispersion relation for any direction with positive and negative field

def dispersion_comparison_negB_plot(k_dict, freq_dict, magnon_density_dict, shading='gouraud'):
    print("Plotting...")

    rasterized = True

    fig = plt.figure(figsize=(mpl_conf.get_width(1.0), mpl_conf.get_height(0.67)))
    gs = fig.add_gridspec(nrows=1, ncols=4, width_ratios=[4, 4, 4, 0.2], wspace=0.07, bottom=0.21, right=0.85)

    fields = sorted(k_dict.keys())

    axs_dict = dict()
    for i, B in zip(range(len(fields)), fields):
        if B == fields[0]:
            axs_dict[B] = fig.add_subplot(gs[0, i])
        else:
            axs_dict[B] = fig.add_subplot(gs[0, i], sharex=axs_dict[fields[0]], sharey=axs_dict[fields[0]])

        axs_dict[B].xaxis.set_major_locator(ticker.MultipleLocator(base=np.pi/2))
        axs_dict[B].xaxis.set_major_formatter(ticker.FuncFormatter(plot_util.multiple_of_pi_over_2))

        if not gs[0, i].is_first_col():
            axs_dict[B].tick_params(labelleft=False)


    # colorbar axis
    cax = fig.add_subplot(gs[0, -1])

    min_magn_dens = np.inf
    max_magn_dens = - np.inf

    for field in fields:
        magnon_density = magnon_density_dict[field]
        min_magn_dens = min(magnon_density.min(), min_magn_dens)
        max_magn_dens = max(magnon_density.max(), max_magn_dens)

    for field in fields:
        if field == 0:
            axs_dict[field].set_title(r"$B = 0$")
        elif field < 0:
            axs_dict[field].set_title(r"$B < 0$")
        else:
            axs_dict[field].set_title(r"$B > 0$")

    im_list = []

    print("[", end="")
    for field in fields:
        print("-", end="")
        ax = axs_dict[field]
        k_vectors = k_dict[field]
        freqs = freq_dict[field]

        freqs *= 1e-15
        if field == fields[0]:
            ax.set_ylabel(r"$\omega$ (\SI{e15}{\radian\per\second})")

        magnon_density = magnon_density_dict[field]
        im = ax.pcolormesh(k_vectors, freqs, magnon_density, shading=shading,
                           norm=colors.LogNorm(vmin=min_magn_dens, vmax=max_magn_dens),
                           rasterized=rasterized)
        im_list.append(im)

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")

        ax.set_xlabel(r"$k \cdot \tilde{a}$")

    print("-", end="")
    cb = fig.colorbar(im_list[-1], cax=cax, orientation="vertical")
    cb.set_label(r"$n(\omega, k)$ (arb. unit)")
    print("]")

    print("Saving fig...")
    fig.savefig(f"{save_base_path}dispersion_comparison_negBfield_{shading[:4]}.pdf")

    # print("Showing fig...")
    # plt.show()



def dispersion_comparison_negB(shading='gouraud'):
    print("\n\nDISPERSION RELATION COMPARISON: POSITIVE AND NEGATIVE MAGNETIC FIELD FIELD")

    paths = {
        -100: "/data/scc/marian.gunsch/10/AM_tilt_Tstairs_T2_x_Bn100-2/",
        0: "/data/scc/marian.gunsch/10/AM_tilt_Tstairs_T2_x-2/",
        100: "/data/scc/marian.gunsch/10/AM_tilt_Tstairs_T2_x_B100-2/"
    }

    dx = physics.lattice_constant

    data_A, data_B = mag_util.npy_files_from_dict(paths, slice_index=-dispersion_data_points,
                                                  max_rows=dispersion_data_points + 10000)

    k_dict = dict()
    freq_dict = dict()
    magnon_density_dict = dict()

    for Bstrength in paths.keys():
        data_points = min(data_A[Bstrength].shape[0], data_B[Bstrength].shape[0], dispersion_data_points)

        if data_points < dispersion_data_points:
            warnings.warn(f"{Bstrength}: Can only run with {data_points} data points.")

        print(f"{paths[Bstrength]=}")
        print(f"{data_A[Bstrength].shape=}")
        print(f"{data_B[Bstrength].shape=}")

        Sx = physics.magnetization(mag_util.get_component(data_A[Bstrength][:data_points], "x", 0),
                                   mag_util.get_component(data_B[Bstrength][:data_points], "x", 0))
        Sy = physics.magnetization(mag_util.get_component(data_A[Bstrength][:data_points], "y", 0),
                                   mag_util.get_component(data_B[Bstrength][:data_points], "y", 0))
        dt = util.get_time_step(paths[Bstrength])
        k, f, m = physics.dispersion(Sx, Sy, dx, dt)
        k_dict[Bstrength] = k
        freq_dict[Bstrength] = f
        magnon_density_dict[Bstrength] = m

    dispersion_comparison_negB_plot(k_dict, freq_dict, magnon_density_dict, shading)


# %% BOUNDARY EFFECTS

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

    fig, gs = create_figure((mpl_conf.get_width(), mpl_conf.get_width() / 1.26))

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


def handle_config_data_aligned(config_data1, config_data2):
    data1 = spinconf_util.average_z_layers(
        spinconf_util.select_SL_and_component(config_data1, "A", "z") +
        spinconf_util.select_SL_and_component(config_data1, "B", "z")
    )
    # This works because no points with different SL occupy the same position and the positions with no atom have been
    # set to zero
    data2 = spinconf_util.average_z_layers(
        spinconf_util.select_SL_and_component(config_data2, "A", "z") +
        spinconf_util.select_SL_and_component(config_data2, "B", "z")
    )
    average_config = np.mean(np.concatenate((data1, data2), axis=2), axis=2)  # avg over z

    return spinconf_util.average_aligned_data(average_config, "default", True, True)


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


def plot_colormap_tilted(fig, axs, x, y, magnetization, y_label="", vmin_=-0.012, vmax_=0.012):
    X, Y = np.meshgrid(x, y, sparse=True, indexing='xy')
    pcms = []
    for ax in axs:
        pcms.append(ax.pcolormesh(X, Y, magnetization, norm=colors.CenteredNorm(), cmap='RdBu_r'))

    vmin = min(p.get_array().min() for p in pcms)
    vmin = min(vmin, vmin_)
    vmax = max(p.get_array().max() for p in pcms)
    vmax = max(vmax, vmax_)
    for p in pcms:
        p.set_clim(vmin, vmax)

    place_colorbar(fig, axs, pcms[0], 0.02, 0.02, y_label)
    # fig.subplots_adjust(left=0.1)


def plot_colormap_aligned(fig, axs, x_centered, y_centered, magn_centered, x_shifted, y_shifted, magn_shifted,
                          y_label="", vmin=-0.012, vmax=0.012):
    X1, Y1 = np.meshgrid(x_centered, y_centered)
    Z1 = magn_centered
    X2, Y2 = np.meshgrid(x_shifted, y_shifted)
    Z2 = magn_shifted

    def make_diamonds(X, Y, Z, size=2.0):
        polys, values = [], []
        for (cx, cy, val) in zip(X.ravel(), Y.ravel(), Z.ravel()):
            values.append(val)
            poly = [(cx, cy + 0.5 * size),
                    (cx + 0.5 * size, cy),
                    (cx, cy - 0.5 * size),
                    (cx - 0.5 * size, cy)]
            polys.append(poly)
        return polys, values

    polys1, values1 = make_diamonds(X1, Y1, Z1, size=x_centered[1] - x_centered[0])
    polys2, values2 = make_diamonds(X2, Y2, Z2, size=x_shifted[1] - x_shifted[0])

    polys = polys1 + polys2
    values = values1 + values2

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    for ax in axs:
        collection = PolyCollection(polys, array=np.array(values),
                                    cmap="RdBu_r", norm=norm)  # , edgecolors="k", linewidth=0.3
        ax.add_collection(collection)
    place_colorbar(fig, axs, collection, 0.02, 0.02, y_label)


def boundary_effects(temperature=2):
    print("\n\nBOUNDARY EFFECTS")

    profile_suffix = "spin-configs-99-999/mag-profile-99-999.altermagnetA.dat"
    config_suffix = "spin-configs-99-999/spin-config-99-999-005000.dat"

    paths_T2 = {
        "100": "/data/scc/marian.gunsch/12/AM_Tstairs_x_T2_openbou/",
        "010": "/data/scc/marian.gunsch/12/AM_Tstairs_y_T2_openbou/",
        "1-10": "/data/scc/marian.gunsch/04/04_AM_tilted_Tstairs_T2_openbou/",
        "110": "/data/scc/marian.gunsch/04/04_AM_tilted_yTstairs_T2_openbou/"
    }
    paths_T0 = {
        "100": "/data/scc/marian.gunsch/12/AM_Tstairs_x_T0_openbou/",
        "010": "/data/scc/marian.gunsch/12/AM_Tstairs_y_T0_openbou/",
        "1-10": "/data/scc/marian.gunsch/12/AM_tilt_Tstairs_x_T0_openbou/",
        "110": "/data/scc/marian.gunsch/12/AM_tilt_Tstairs_y_T0_openbou/"
    }
    if temperature == 2:
        paths = paths_T2
    elif temperature == 0:
        paths = paths_T0
    else:
        raise ValueError(f"No paths available for a temperature of {temperature} meV.")

    tilted_dict = {
        "100": False,
        "010": False,
        "1-10": True,
        "110": True
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

        real_space[direction] = np.arange(0.5, magnetization_profile[direction].shape[0], 1.0) if tilted_dict[direction] \
            else np.arange(0.0, magnetization_profile[direction].shape[0], 1.0)
        real_space[direction] = physics.index_to_position(real_space[direction], tilted_dict[direction])

    # Configuration data
    config_data = dict()
    for direction in paths:
        config_data[direction] = spinconf_util.read_spin_config_dat(f"{paths[direction]}{config_suffix}",
                                                                    is_tilted=tilted_dict[direction],
                                                                    fixed_version=True, empty_value=0.0)

    # TILTED
    print("Plotting tilted...")
    magn_config = handle_config_data_tilted(config_data["1-10"], config_data["110"])

    fig_tilted, *axs_tilted = broken_axes_boundary_plot(
        real_space["1-10"], magnetization_profile["1-10"],
        real_space["110"], magnetization_profile["110"],
        0, real_space["110"][-1] + physics.index_to_position(0.5, True),
        physics.index_to_position(8.2, True),
        r"position $x/a$ in direction \hkl[1-10]",
        r"position $y/a$ in direction \hkl[110]",
        r"$\langle S^z \rangle$"
    )

    plot_colormap_tilted(fig_tilted, axs_tilted, real_space["1-10"], real_space["110"], magn_config,
                         r"$\langle S^z \rangle$")

    fig_tilted.savefig(f"out/thesis/equilibrium/boundary_tilted_T{temperature}.pdf")

    plt.show()

    # ALIGNED
    print("Plotting aligned...")
    x_centered, y_centered, magn_centered, x_shifted, y_shifted, magn_shifted = handle_config_data_aligned(
        config_data["100"], config_data["010"])

    fig_aligned, *axs_aligned = broken_axes_boundary_plot(real_space["100"], magnetization_profile["100"],
                                                          real_space["010"], magnetization_profile["010"],
                                                          physics.index_to_position(-0.5, False),
                                                          real_space["100"][-1] + physics.index_to_position(0.5, False),
                                                          physics.index_to_position(8.2, True),
                                                          r"position $x/a$ in direction \hkl[100]",
                                                          r"position $y/a$ in direction \hkl[010]",
                                                          r"$\langle S^z \rangle$")

    plot_colormap_aligned(fig_aligned, axs_aligned,
                          x_centered, y_centered, magn_centered, x_shifted, y_shifted, magn_shifted,
                          r"$\langle S^z \rangle$")

    fig_aligned.savefig(f"out/thesis/equilibrium/boundary_aligned_T{temperature}.pdf")

    plt.show()


# %% Main

def main():
    pass
    boundary_effects(2)
    boundary_effects(0)

    equilibrium_comparison_Bfield()

    # dispersion_comparison_Bfield_table(2)

    # dispersion_comparison_negB()
