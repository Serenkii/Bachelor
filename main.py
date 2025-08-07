import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import scipy as sp
from scipy.optimize import curve_fit

import src.utility as util
import src.mag_util as mag_util
import src.bulk_util as bulk_util
import src.utility as util
import src.plot_util as plot_util
import src.physics as physics
import src.spinconf_util as spinconf_util
import src.helper as helper

seperator = "-------------------------------------------------------------\n"

plot_paths = True

# %% configure matplotlib
# mpl.use('Qt5Agg')   # for interactive plots https://stackoverflow.com/questions/49844189/how-to-get-interactive-plot-of-pyplot-when-using-pycharm
# See here: https://matplotlib.org/stable/users/explain/figure/backends.html

# %% Meeting in May

def temperature_dependent_nernst(save=False, save_path='out/T-dependent-nernst.png', delta_x=0):
    """
    Analysis of the temperature dependent nernst effect, without DMI, used job-script:
    'job-AM_tiltX_Tstep_zlayers_noABC_differentTemps.sh'
    :param save:
    :param save_path:
    :param delta_x:
    :return:
    """

    # function does not work here, because not same amount of lines
    # data_dict_A, data_dict_B, npz_path = mag_util.save_arrayjob_as_npz(
    #     "/data/scc/marian.gunsch/AM_tiltedX_Tstep_nernst_T",
    #     "data/nernst/arrj_nernst_Tdependent",
    #     10)

    print("Running analysis of the temperature dependent nernst effect. No DMI. "
          "Used jobscript: 'job-AM_tiltX_Tstep_zlayers_noABC_differentTemps.sh', "
          "data_path='/data/scc/marian.gunsch/AM_tiltedX_Tstep_nernst_T'")

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
        magn[key] = physics.magnetization(Sz_A[key], Sz_B[key])
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
    ax1.legend(loc='center', ncols=2)
    # ax2.legend()
    if save:
        print(f"Saving to '{save_path}'")
        plt.savefig(save_path)
    plt.show()

    # temperature dependent magnitude
    boundary_size = 50

    temperatures = np.array(list(data_dict_A.keys()), dtype=int)
    neel_magnitudes = np.empty(temperatures.shape, dtype=float)
    magn_magnitudes = np.empty(temperatures.shape, dtype=float)

    for i in range(temperatures.shape[0]):
        neel_magnitudes[i] = np.average(neel[str(temperatures[i])][boundary_size:-boundary_size])
        magn_magnitudes[i] = np.average(magn[str(temperatures[i])][boundary_size:-boundary_size])

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle("Temperature dependence of nernst effect (magnitudes)")
    ax1.set_title("Average Magnitude of Neel-Vector")
    ax2.set_title("Average Magnitude of Magnetization")
    ax2.set_xlabel("Temperature in meV")
    ax1.set_ylabel("magnitude (au)")
    ax2.set_ylabel("magnitude (au)")

    ax1.plot(temperatures, np.abs(neel_magnitudes), marker='o', linestyle='')
    ax2.plot(temperatures, np.abs(magn_magnitudes), marker='o', linestyle='')

    if save:
        print(f"Saving to '{save_path[:-4]}_Tplot.png'")
        plt.savefig(f"{save_path[:-4]}_Tplot.png")
    plt.show()



def dmi_ground_state_comparison(save=False, save_path='out/T-dependent-nernst.png'):
    equi_path = "/data/scc/marian.gunsch/AM_tiltedX_ttmstairs_T2meV/AM_Teq2meV-99-999.dat"
    equi_dmi_path = "/data/scc/marian.gunsch/AM_tiltedX_ttmstairs_DMI-2/AM_Teq-99-999.dat"  # this seems weird!!!
    print("CAREFUL!\n"
          f"{equi_path} seems to have faulty data! Maybe it was never able to reach equilibrium because no start config"
          f" ferri was used?\n"
          f"Use bulk_util.p({equi_dmi_path}) to see the weirdness when equilibriating...")
    equi_dmi_path = "/data/scc/marian.gunsch/AM_tiltedX_ttmstairs_DMI_ferri-2/AM_Teq-99-999.dat"
    print(f"\tNow using {equi_dmi_path} instead...")

    data_noDMI = np.loadtxt(equi_path)
    data_DMI = np.loadtxt(equi_dmi_path)

    spins_A = np.average(bulk_util.get_components(data_noDMI, 'A', 'xyz', 100), axis=0)
    spins_B = np.average(bulk_util.get_components(data_noDMI, 'B', 'xyz', 100), axis=0)
    spins_DMI_A = np.average(bulk_util.get_components(data_DMI, 'A', 'xyz', 100), axis=0)
    spins_DMI_B = np.average(bulk_util.get_components(data_DMI, 'B', 'xyz', 100), axis=0)

    print(f"Spins A: {spins_A}")
    print(f"Spins B: {spins_B}")
    print(f"Spins with DMI A: {spins_DMI_A}")
    print(f"Spins with DMI B: {spins_DMI_B}")

    # Example data: 4 types of values per group
    types = ['SL A, no DMI', 'SL B, no DMI', 'SL A, DMI', 'SL B, DMI']

    x_values = [spins_A[0], spins_B[0], spins_DMI_A[0], spins_DMI_B[0]]
    y_values = [spins_A[1], spins_B[1], spins_DMI_A[1], spins_DMI_B[1]]
    z_values = [spins_A[2], spins_B[2], spins_DMI_A[2], spins_DMI_B[2]]

    # Combine them by type
    data = {
        'x': x_values,
        'y': y_values,
        'z': z_values
    }

    # Prepare plot
    fig, ax = plt.subplots(figsize=(7, 5))

    # Assign a color or marker for each type
    markers = ['^', 'o', '^', 'o']  # Circle, square, triangle, diamond
    colors = ['red', 'orange', 'blue', 'purple']

    # Plot each type across all groups (x, y, z)
    for i, t in enumerate(types):
        group_labels = list(data.keys())  # ['x', 'y', 'z']
        values = [data[label][i] for label in group_labels]  # Get i-th value for each group
        ax.scatter(group_labels, values, label=f'{t}', marker=markers[i], color=colors[i], alpha=0.7)

    # Labeling
    ax.set_ylabel('Average value')
    ax.set_title('Equilibrium 2meV: Spin components for different sublattices with and without DMI')
    ax.legend()

    if save:
        print(f"Saving to {save_path[:-4]}_1.pdf")
        fig.savefig(f"{save_path[:-4]}_1.pdf")

    plt.show()

    ## Neel and Magnetization
    neel = []
    neel_DMI = []
    magn = []
    magn_DMI = []

    for i in range(3):
        neel.append(physics.neel_vector(spins_A[i], spins_B[i]))
        neel_DMI.append(physics.neel_vector(spins_DMI_A[i], spins_DMI_B[i]))
        magn.append(physics.magnetization(spins_A[i], spins_B[i]))
        magn_DMI.append(physics.magnetization(spins_DMI_A[i], spins_DMI_B[i]))



    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Equilibrium 2meV: Neel and Magn with and without DMI")
    ax1.set_title('Magnetization')
    ax1.set_ylabel('Magnitude (au)')
    ax2.set_title('Neel-Vector')
    # ax2.set_ylabel('Magnitude (au)')

    label_list = ['no DMI', 'DMI']

    ax1.plot(label_list, [magn, magn_DMI], linestyle='', marker='o', label=['x', 'y', 'z'])
    ax2.plot(label_list, [neel, neel_DMI], linestyle='', marker='o')

    fig.legend()

    if save:
        print(f"Saving to 1{save_path[:-4]}_2.pdf")
        fig.savefig(f"{save_path[:-4]}_2.pdf")

    plt.show()

    print("\n")


def dmi_comparison(dmi_path_A, dmi_path_B, no_dmi_path_A, no_dmi_path_B, title="Comparison DMI", delta_x=0,
                   save_path=None):
    data_dmi_A = np.loadtxt(dmi_path_A)
    data_dmi_B = np.loadtxt(dmi_path_B)
    data_nodmi_A = np.loadtxt(no_dmi_path_A)
    data_nodmi_B = np.loadtxt(no_dmi_path_B)

    Sz_dmi_A = util.time_avg(mag_util.get_component(data_dmi_A, 'z', 1))
    Sz_dmi_B = util.time_avg(mag_util.get_component(data_dmi_B, 'z', 1))
    Sz_nodmi_A = util.time_avg(mag_util.get_component(data_nodmi_A, 'z', 1))
    Sz_nodmi_B = util.time_avg(mag_util.get_component(data_nodmi_B, 'z', 1))

    neel_dmi = physics.neel_vector(Sz_dmi_A, Sz_dmi_B)
    magn_dmi = physics.magnetization(Sz_dmi_A, Sz_dmi_B)

    neel_nodmi = physics.neel_vector(Sz_nodmi_A, Sz_nodmi_B)
    magn_nodmi = physics.magnetization(Sz_nodmi_A, Sz_nodmi_B)

    x = np.arange(delta_x, neel_dmi.size - delta_x, 1.0)

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle(title)
    ax1.set_title("Neel-Vector")
    ax2.set_title("Magnetization")
    ax1.set_ylabel("magnitude (au)")
    ax2.set_ylabel("magnitude (au)")
    # ax1.set_xlabel('Grid position')
    ax2.set_xlabel('Grid position')

    if delta_x > 0:
        ax1.plot(x, neel_dmi[delta_x:-delta_x], label="DMI")
        ax1.plot(x, neel_nodmi[delta_x:-delta_x], label="no DMI")
        ax2.plot(x, magn_dmi[delta_x:-delta_x], label="DMI")
        ax2.plot(x, magn_nodmi[delta_x:-delta_x], label="no DMI")
    else:
        ax1.plot(x, neel_dmi, label="DMI")
        ax1.plot(x, neel_nodmi, label="no DMI")
        ax2.plot(x, magn_dmi, label="DMI")
        ax2.plot(x, magn_nodmi, label="no DMI")

    ax1.legend()
    ax2.legend()

    if save_path:
        print(f"Saving to {save_path}")
        fig.savefig(f"{save_path}")

    plt.show()

    return magn_nodmi, magn_dmi, neel_nodmi, neel_dmi



def quick_seebeck_dmi_comparison():
    print("Comparing the spin Seebeck effect with and without antisymmetric exchange at a temperature of 2 meV.")
    dmi_path_A = "/data/scc/marian.gunsch/AM-DMI_tilted_Tstep_seebeck/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat"
    dmi_path_B = "/data/scc/marian.gunsch/AM-DMI_tilted_Tstep_seebeck/spin-configs-99-999/mag-profile-99-999.altermagnetB.dat"
    no_dmi_path_A = "/data/scc/marian.gunsch/AM-tilted_Tstep_seebeck/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat"
    no_dmi_path_B = "/data/scc/marian.gunsch/AM-tilted_Tstep_seebeck/spin-configs-99-999/mag-profile-99-999.altermagnetB.dat"
    delta_x = 10


    magn_nodmi_, magn_dmi_, _, _ = dmi_comparison(
        dmi_path_A,
        dmi_path_B,
        no_dmi_path_A,
        no_dmi_path_B,
        "Comparison Seebeck (no equilibriums were subtracted!)",
        delta_x,
        "out/comparison_DMI_seebeck_.pdf"
    )

    # Compare maximum values

    max_nodmi = np.max(magn_nodmi_)
    max_dmi = np.max(magn_dmi_)

    fig, ax1 = plt.subplots()
    fig.suptitle("Comparison: Maximum value of magnetization for SSE with(out) DMI")
    ax1.set_title('Magnetization')
    ax1.set_ylabel('Magnitude (au)')

    label_list = ['no DMI', 'DMI']

    ax1.plot(label_list, [max_nodmi, max_dmi], linestyle='', marker='o')

    print("Saving to out/max_value_SSE_DMI_comparison.pdf")
    fig.savefig(f"out/max_value_SSE_DMI_comparison.pdf")

    plt.show()

    ####


    save_path = None
    save_path = "out/comparison_DMI_magnon_accumulation_.pdf"

    rel_step_pos = 0.49

    data_dmi_A = np.loadtxt(dmi_path_A)
    data_dmi_B = np.loadtxt(dmi_path_B)
    data_nodmi_A = np.loadtxt(no_dmi_path_A)
    data_nodmi_B = np.loadtxt(no_dmi_path_B)

    data_nodmi_eq = np.loadtxt("/data/scc/marian.gunsch/AM_tiltedX_ttmstairs_T2meV/AM_Teq2meV-99-999.dat")
    data_dmi_eq = np.load("data/DMI/seebeck/equi_T2meV_DMI5.npy")

    _, _, magnon_acc_dmi, delta_neel_dmi = physics.seebeck(
        data_dmi_A,
        data_dmi_B,
        data_dmi_eq,
        rel_step_pos
    )
    _, _, magnon_acc_nodmi, delta_neel_nodmi = physics.seebeck(
        data_nodmi_A,
        data_nodmi_B,
        data_nodmi_eq,
        rel_step_pos
    )

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle("Comparison of magnon accumulation and delta neel with(out) DMI")
    ax1.set_title('delta neel (equi subtracted)')
    ax2.set_title('Magnon accumulation')
    ax1.set_ylabel('Magnitude (au)')
    ax2.set_ylabel('Magnitude (au)')
    ax2.set_xlabel('Grid position')

    x = np.arange(delta_x, delta_neel_dmi.size - delta_x, 1.0)

    if delta_x > 0:
        ax1.plot(x, delta_neel_dmi[delta_x:-delta_x], label="DMI")
        ax1.plot(x, delta_neel_nodmi[delta_x:-delta_x], label="no DMI")
        ax2.plot(x, magnon_acc_dmi[delta_x:-delta_x], label="DMI")
        ax2.plot(x, magnon_acc_nodmi[delta_x:-delta_x], label="no DMI")
    else:
        ax1.plot(x, delta_neel_dmi, label="DMI")
        ax1.plot(x, delta_neel_nodmi, label="no DMI")
        ax2.plot(x, magnon_acc_dmi, label="DMI")
        ax2.plot(x, magnon_acc_nodmi, label="no DMI")

    ax1.legend()
    ax2.legend()

    if save_path:
        print(f"Saving to {save_path}")
        fig.savefig(f"{save_path}")

    plt.show()


def quick_nernst_dmi_comparison():
    print("Comparing the spin Nernst effect with and without antisymmetric exchange at a temperature of 2 meV. "
          "It has to be noted that only the minimum amount of z-layers of 2 were used. That's why there was not much to"
          "average and the result may be very noisy. Hopefully one can still qualitatively compare with and without "
          "DMI")

    dmi_comparison(
        "/data/scc/marian.gunsch/AM-DMI_tilted_Tstep_nernst/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat",
        "/data/scc/marian.gunsch/AM-DMI_tilted_Tstep_nernst/spin-configs-99-999/mag-profile-99-999.altermagnetB.dat",
        "/data/scc/marian.gunsch/AM-tilted_Tstep_nernst/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat",
        "/data/scc/marian.gunsch/AM-tilted_Tstep_nernst/spin-configs-99-999/mag-profile-99-999.altermagnetB.dat",
        "Comparison Nernst",
        10,
        "out/comparison_DMI_nernst_.pdf"
    )


def presenting_data_01():
    temperature_dependent_nernst(save=True, save_path='out/T-dependent-nernst.png', delta_x=0)
    dmi_ground_state_comparison(save=True, save_path='out/ground_state_comparison_DMI.pdf')

    quick_seebeck_dmi_comparison()
    quick_nernst_dmi_comparison()


# %% Meeting in June

def magnetization_neel_2d_plot(path_conf, eq_path_conf, temperature=2, save_prefix=None):
    print("MAGNETIZATION AND NEEL VECTOR\n")

    path = path_conf
    data = spinconf_util.read_spin_config_dat(path)

    eq_path = eq_path_conf
    eq_data = spinconf_util.read_spin_config_dat(eq_path)

    print(f"Finished reading data...\n"
          f"Data path: {path}\n"
          f"Equilibrium path: {eq_path}\n")

    magn, neel = spinconf_util.calculate_magnetization_neel(data, eq_data, rel_Tstep_pos=0.49)

    magn_zavg = spinconf_util.average_z_layers(magn["z"])
    neel_zavg = spinconf_util.average_z_layers(neel["z"])

    if not save_prefix:
        spinconf_util.plot_colormap(magn_zavg, title=f"magnetization, equi subtracted (T={temperature}meV)",
                                    fig_comment=f"{path_conf} \n - {eq_path}")
        spinconf_util.plot_colormap(neel_zavg, title=f"Neel vector (z), equi subtracted (T={temperature}meV)",
                                    fig_comment=f"{path_conf} \n - {eq_path}")

        spinconf_util.plot_colormap(spinconf_util.convolute(magn_zavg),
                                    title=f"magnetization, equi subtracted (T={temperature}meV) - convoluted",
                                    fig_comment=f"{path_conf} \n - {eq_path}")
        spinconf_util.plot_colormap(spinconf_util.convolute(neel_zavg),
                                    title=f"Neel vector (z), equi subtracted (T={temperature}meV) - convoluted",
                                    fig_comment=f"{path_conf} \n - {eq_path}")
    else:
        spinconf_util.plot_colormap(magn_zavg, title=f"magnetization, equi subtracted (T={temperature}meV)",
                                    save_path=f"{save_prefix}_magn_equisubtr.pdf",
                                    fig_comment=f"{path_conf} \n - {eq_path}")
        spinconf_util.plot_colormap(neel_zavg, title=f"Neel vector (z), equi subtracted (T={temperature}meV)",
                                    save_path=f"{save_prefix}_neel_equisubtr.pdf",
                                    fig_comment=f"{path_conf} \n - {eq_path}")

        spinconf_util.plot_colormap(spinconf_util.convolute(magn_zavg),
                                    title=f"magnetization, equi subtracted (T={temperature}meV) - convoluted",
                                    save_path=f"{save_prefix}_magn_conv_equisubtr.pdf",
                                    fig_comment=f"{path_conf} \n - {eq_path}")
        spinconf_util.plot_colormap(spinconf_util.convolute(neel_zavg),
                                    title=f"Neel vector (z), equi subtracted (T={temperature}meV) - convoluted",
                                    save_path=f"{save_prefix}_neel_conv_equisubtr.pdf",
                                    fig_comment=f"{path_conf} \n - {eq_path}")


def spin_currents_2d_plot(data_path, save_prefix=None, show_path=True):
    print("SPIN CURRENTS\n"
          "I attempted to gain useful data by calculating the spin currents. Luckily, you can clearly see the Spin "
          "Seebeck effect when looking at the longitudinal direction of the spin currents. Sadly, for Spin Nernst, "
          "not much is visible. The following will be plotted: 1. spin currents for SSE, 2. spin currents for SNE, "
          "3. spin currents for SNE but convoluted.")

    data = spinconf_util.read_spin_config_dat(data_path)

    print(f"Finished reading data from {data_path}...")

    directions = ["longitudinal", "transversal", "transversal"]
    convolutions = ["none", "none", "denoise"]

    for direction, convolution in zip(directions, convolutions):
        *data_tuple, j_other_paper = spinconf_util.average_z_layers(
            *spinconf_util.calculate_spin_currents(data, direction)
        )

        for i in range(len(data_tuple)):
            data_tuple[i] = spinconf_util.convolute(data_tuple[i], convolution,
                                                    denoise_kwargs=dict(sigma_color=0.01, sigma_spatial=4, mode='edge'))


        X, Y = np.meshgrid(np.arange(0, data_tuple[0].shape[1], 1, dtype=int),
                           np.arange(0, data_tuple[0].shape[0], 1, dtype=int),
                           sparse=True, indexing="xy")

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        axs = axs.flatten()
        titles = ["j_inter_p", "j_inter_m", "j_intra_A", "j_intra_B"]

        for i in range(len(titles)):
            titles[i] = f"{titles[i]}, {direction}, {convolution}"

        for ax, data_grid, title in zip(axs, data_tuple, titles):
            im = ax.pcolormesh(X, Y, data_grid, norm=colors.CenteredNorm(), cmap='RdBu_r')
            ax.set_title(title)
            ax.set_aspect('equal', 'box')
            ax.margins(x=0, y=0)
            fig.colorbar(im, ax=ax)

        fig.tight_layout()

        if show_path:
            fig.text(0.5, 0.5, data_path, color="green", ha="center", va="center")

        if save_prefix:
            save_path = f"{save_prefix}_{direction[:4]}_conv-{convolution}.pdf"
            print(f"Saving to {save_path}...")
            fig.savefig(save_path)

        plt.show()

        print(f"Plotting spin currents in {direction} direction with convolution '{convolution}'...")

        # TODO: print statement on what was plotted and saving figure


def position_dependent_frequency_spectrum(data_A_path, data_B_path=None, save_prefix=None, show_path=True):
    print("MAGNON DENSITY AS A FUNCTION OF FREQUENCY\n"
          "For the spin Nernst effect, we want to look at the frequency dependency of the magnon density as a function "
          "in space. We want to know the magnon density for different frequencies at different positions along the "
          "orthogonal direction of the temperature gradient (step). Therefore we want a two-dimensional colorplot, one "
          "axis is the position and the other is the frequency. The color then corresponds to the magnon density.\n"
          "The procedure is not optimal. Ideally we would take all (for each grid position) Sx and Sy, Fourier "
          "transform, square them and "
          "THEN average. This is not possible because spin configurations don't include all time steps. Therefore "
          "we now try to take the magnetic profile (where all the spins were already averaged) and then we square."
          )

    # 001 boundaries
    # data_A_path = "/data/scc/marian.gunsch/01_AM_tilted_Tstep/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat"
    # data_B_path = "/data/scc/marian.gunsch/01_AM_tilted_Tstep/spin-configs-99-999/mag-profile-99-999.altermagnetB.dat"
    #
    # data_A_path = "/data/scc/marian.gunsch/02_AM_tilted_Tstep_hightres/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat"
    # data_B_path = "/data/scc/marian.gunsch/02_AM_tilted_Tstep_hightres/spin-configs-99-999/mag-profile-99-999.altermagnetB.dat"

    if not data_B_path:
        data_B_path = f"{data_A_path[:-5]}B.dat"

    data_A = np.loadtxt(data_A_path)
    data_B = np.loadtxt(data_B_path)

    print(f"Finished reading data from {data_A_path}")

    skip_time_steps = -100000
    # skip_time_steps = -50000

    Sx = mag_util.get_component(data_A, "x", skip_time_steps) + mag_util.get_component(data_B, "x", skip_time_steps)
    Sx *= 0.5
    Sy = mag_util.get_component(data_A, "y", skip_time_steps) + mag_util.get_component(data_B, "y", skip_time_steps)
    Sy *= 0.5

    Sp = Sx + 1j * Sy
    Sm = Sx - 1j * Sy

    # Fourier
    time_steps = mag_util.get_component(data_A, "t", skip_time_steps)
    dt = time_steps[1] - time_steps[0]
    dt *= 1e-16     # unit: picoseconds

    Sp_F_ = np.fft.fft(Sp, axis=0)
    Sm_F_ = np.fft.fft(Sm, axis=0)

    freqs = np.fft.fftfreq(time_steps.shape[0], d=dt)
    omega = 2 * np.pi * freqs

    Sp_F = np.fft.fftshift(Sp_F_, axes=0)
    Sm_F = np.fft.fftshift(Sm_F_, axes=0)
    omega_shifted = np.fft.fftshift(omega)

    # magnon_density = np.abs(Sp_F * Sm_F) # ** 2      # Should this be only real (without np.abs)?
    # magnon_density = Sp_F * Sm_F
    magnon_density = np.abs(Sp_F) ** 2

    print("Plotting...")

    # plotting
    fig, ax = plt.subplots()
    ax.set_title("Position and frequency dependent magnon density")
    positions = np.arange(0, Sx.shape[1], 1)
    # im = ax.pcolormesh(positions[:60], omega_shifted, magnon_density[:,:60], shading='auto')
    im = ax.pcolormesh(positions, omega_shifted, magnon_density, shading='auto', norm=colors.LogNorm(vmin=magnon_density.min(), vmax=magnon_density.max()))
    # im = ax.pcolormesh(positions, omega_shifted, magnon_density, shading='auto')
    # im = ax.pcolormesh(positions[:30], omega_shifted[9000:-9000], magnon_density[9000:-9000,:30], shading='auto', norm=colors.LogNorm(vmin=magnon_density.min(), vmax=magnon_density.max()))
    ax.set_ylabel('Frequency ω in rad/s')
    ax.set_xlabel('Position (index)')
    fig.colorbar(im, ax=ax, label='Magnon Density')

    if show_path:
        fig.text(0.5, 1.0, f"{data_A_path}", ha="center", va="top", color="green", size=6)

    if save_prefix:
        fig.savefig(f"{save_prefix}_colormesh.pdf")

    plt.show()

    for position in [0, 1, 2, 10, 50, -1]:
        fig, ax = plt.subplots()
        ax.set_title(f"Spectrum at position index {position}")
        ax.set_xlabel("Frequency ω in rad/s")
        ax.set_ylabel("Amplitude (magnon density)")
        ax.plot(omega_shifted, magnon_density[:, position], linewidth=0.3, label=f"position {position}")
        # ax.set_ylim(bottom=20, top=40)
        # ax.set_xlim(left=-0.5, right=0.5)

        if show_path:
            fig.text(0.5, 1.0, f"{data_A_path}", ha="center", va="top", color="green", size=6)

        if save_prefix:
            fig.savefig(f"{save_prefix}_i{position}.pdf")

        plt.show()


def presenting_data_02():
    seperator = "-------------------------------------------------------------\n"
    print("[05.06.25] Presenting data for next meeting with Uli. We are talking about/showing the following:\n"
          "We want to show the 2D-plots for the whole spin configuration. We want to show SNE, SSE and the Fourier "
          "analysis.\n"
          + seperator)

    out_prefix = "out/02_nernst_and_more/"

    position_dependent_frequency_spectrum(
        "/data/scc/marian.gunsch/02_AM_tilted_Tstep_hightres/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat",
        save_prefix=f"{out_prefix}magnon_acc_frequency")
    print(seperator)

    return

    print("Showing magnetizazion and Neel vector for a temperature of T=2meV. We are subtracting the equilibrium state."
          " One plot with and one without convolution.\n"
          "Parameters of the simulation: open open open boundaries, 512x512x64")
    magnetization_neel_2d_plot(
        "/data/scc/marian.gunsch/AM_tiltedX_Tstep_nernst_T2/spin-configs-99-999/spin-config-99-999-005000.dat",
        "/data/scc/marian.gunsch/AM_tiltedX_ttmstairs_T2meV/spin-configs-99-999/spin-config-99-999-005000.dat",
        2, f"{out_prefix}spin_config_T2step")
    print(seperator)

    # No info for SNE because of 010 boundaries (periodic destroys any edge effects)
    # magnetization_neel_2d_plot(
    #     "/data/scc/marian.gunsch/04_AM_tilted_xTstep_T2/spin-configs-99-999/spin-config-99-999-005000.dat",
    #     "/data/scc/marian.gunsch/04_AM_tilted_Tstairs_T2/spin-configs-99-999/spin-config-99-999-005000.dat")
    # print(seperator)
    print(seperator)

    spin_currents_2d_plot(
        "/data/scc/marian.gunsch/AM_tiltedX_ttmstep_morezlayers_noABC/spin-configs-99-999/spin-config-99-999-005000.dat",
        f"{out_prefix}config_spin_currents_T7step")
    print(seperator)

    spin_currents_2d_plot(
        "/data/scc/marian.gunsch/01_AM_tilted_Tstep/spin-configs-99-999/spin-config-99-999-005000.dat",
        f"{out_prefix}config_spin_currents_T2step")
    print(seperator)

    spin_currents_2d_plot(
        "/data/scc/marian.gunsch/04_AM_tilted_xTstep_T2/spin-configs-99-999/spin-config-99-999-005000.dat",
        f"{out_prefix}config_spin_currents_T2step_morez")
    print(seperator)


    # fourier_thingy_TODO_CHANGENAME()
    # print(seperator)


# %% Further stuff for their paper (03)

def seebeck_03(file_path_quantity, file_path_quantity_eq_subtracted):
    print()

    noDMI_kwargs = dict(alpha=0.7, linestyle="--", linewidth=1.0)

    prefix = "/data/scc/marian.gunsch/"
    mag_util.plot_magnetic_profile_from_paths(
        [f"{prefix}03_AM_tilted_xTstep_DMI/spin-configs-99-999/mag-profile-99-999.altermagnet",
         f"{prefix}03_AM_tilted_yTstep_DMI/spin-configs-99-999/mag-profile-99-999.altermagnet",
         f"{prefix}AM_mag-accumu_tilt_x-axis_kT-7/spin-configs-99-999/mag-profile-99-999.altermagnet",
         f"{prefix}AM_mag-accumu_tilt_y-axis_kT-7/spin-configs-99-999/mag-profile-99-999.altermagnet"], [0, 131, 0, 0],
        file_path_quantity, None, None, 0.49, [dict(label="DMI, [110]"),
                                               dict(label="DMI, [-110]"),
                                               dict(label="no DMI, [110]", **noDMI_kwargs),
                                               dict(label="no DMI, [-110]", **noDMI_kwargs)], "(T=7meV)")

    # Subtracting equilibrium
    spins_warm_A, spins_warm_B = mag_util.get_mean(
        "/data/scc/marian.gunsch/03_AM_tilted_Tstairs_DMI/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat",
        "/data/scc/marian.gunsch/03_AM_tilted_Tstairs_DMI/spin-configs-99-999/mag-profile-99-999.altermagnetB.dat"
    )
    spins_cold_A, spins_cold_B = mag_util.get_mean(
        "/data/scc/marian.gunsch/03_AM_tilted_Tstairs_DMI_T0/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat",
        "/data/scc/marian.gunsch/03_AM_tilted_Tstairs_DMI_T0/spin-configs-99-999/mag-profile-99-999.altermagnetB.dat"
    )
    spins_warm_nodmi_A, spins_warm_nodmi_B = bulk_util.get_mean("data/temp/altermagnet-equilibrium-7meV.dat")
    for component in spins_warm_nodmi_A.keys():
        spins_warm_nodmi_A[component] *= -1
        spins_warm_nodmi_B[component] *= -1        # for some reason here the spin values (z) for A are negative and for B are positive...

    mag_util.plot_magnetic_profile_from_paths(
        [f"{prefix}03_AM_tilted_xTstep_DMI/spin-configs-99-999/mag-profile-99-999.altermagnet",
         f"{prefix}03_AM_tilted_yTstep_DMI/spin-configs-99-999/mag-profile-99-999.altermagnet",
         f"{prefix}AM_mag-accumu_tilt_x-axis_kT-7/spin-configs-99-999/mag-profile-99-999.altermagnet",
         f"{prefix}AM_mag-accumu_tilt_y-axis_kT-7/spin-configs-99-999/mag-profile-99-999.altermagnet"], [0, 131, 0, 0],
        file_path_quantity_eq_subtracted, [(spins_warm_A, spins_warm_B),
                                           (spins_warm_A, spins_warm_B),
                                           (spins_warm_nodmi_A, spins_warm_nodmi_B),
                                           (spins_warm_nodmi_A, spins_warm_nodmi_B)], [(spins_cold_A, spins_cold_B),
                                                                                       (spins_cold_A, spins_cold_B),
                                                                                       (dict(x=0, y=0, z=1),
                                                                                        dict(x=0, y=0, z=-1)),
                                                                                       # maybe 1 and -1 swapped
                                                                                       (dict(x=0, y=0, z=1),
                                                                                        dict(x=0, y=0, z=-1))], 0.49,
        [dict(label="DMI, [110]"),
         dict(label="DMI, [-110]"),
         dict(label="no DMI, [110]", **noDMI_kwargs),
         dict(label="no DMI, [-110]", **noDMI_kwargs)], "(T=7meV, equi subtracted)")





def plot_2d(load_path, save_path=None, width_xy=100, title_suffix=""):
    print()

    spinconf_data = load_path

    equi_data = spinconf_util.average_z_layers(spinconf_util.read_spin_config_dat(spinconf_data))

    print(f"Finished reading equilibrium data from '{spinconf_data}'...")

    magnetization = dict()
    for component in "xyz":
        magnetization[component] = np.squeeze(
            physics.magnetization(spinconf_util.select_SL_and_component(equi_data, "A", component),
                                  spinconf_util.select_SL_and_component(equi_data, "B", component)))

    for component in magnetization.keys():
        data_grid = magnetization[component]

        X, Y = np.meshgrid(np.arange(0, data_grid.shape[1], 1, dtype=int),
                       np.arange(0, data_grid.shape[0], 1, dtype=int),
                       sparse=True, indexing='xy')

        middle = int(data_grid.shape[0] / 2)
        lower = int(max(middle - width_xy / 2, 0)) - 0.5
        upper = int(min(middle + width_xy / 2, data_grid.shape[0])) - 0.5

        fig, ax = plt.subplots()
        ax.set_xlabel("Grid position in direction [110]")
        ax.set_ylabel("Grid position in direction [-110]")

        ax.set_aspect('equal', 'box')
        ax.set_title(f"Magnetization: {component}-component {title_suffix}")
        im = ax.pcolormesh(X, Y, data_grid, norm=colors.CenteredNorm(), cmap='RdBu_r')
        fig.colorbar(im, ax=ax)

        # ax.margins(x=0, y=0)

        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)

        fig.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}{component}.pdf")

        plt.show()



def presenting_data_03():
    print("For Tobias' paper...")

    print("Seebeck")
    seebeck_03("out/03_tobi_paper/seebeck", "out/03_tobi_paper/seebeck_eq")

    load_paths = [
        "/data/scc/marian.gunsch/03_AM_tilted_Tstairs_DMI/spin-configs-99-999/spin-config-99-999-005000.dat",
        "/data/scc/marian.gunsch/AM_tiltedX_ttmstairs_DMI_ferri-2/spin-configs-99-999/spin-config-99-999-005000.dat",
        "/data/scc/marian.gunsch/AM_tiltedX_ttmstairs_DMI_large_ferri/spin-configs-99-999/spin-config-99-999-005000.dat"
    ]
    save_paths = ["out/03_tobi_paper/2d_plot_T7_",
                  "out/03_tobi_paper/2d_plot_T2_",
                  "out/03_tobi_paper/2d_plot_T2_largeDMI_",]

    print("Equilibrium states")
    for load_path, save_path in zip(load_paths, save_paths):
        print(".")
        plot_2d(load_path, save_path)

    print("\n\nSeebeck 2d\n")
    plot_2d("/data/scc/marian.gunsch/03_AM_tilted_xTstep_DMI/spin-configs-99-999/spin-config-99-999-005000.dat",
            "out/03_tobi_paper/2d_plot_seebeck_110_")
    plot_2d("/data/scc/marian.gunsch/03_AM_tilted_yTstep_DMI/spin-configs-99-999/spin-config-99-999-005000.dat",
            "out/03_tobi_paper/2d_plot_seebeck_-110_")
    plot_2d("/data/scc/marian.gunsch/AM_tiltedX_Tstep_nernst_T2/spin-configs-99-999/spin-config-99-999-005000.dat",
            "out/03_tobi_paper/2d_plot_seebeck_110_T2_noDMI_")
    plot_2d("/data/scc/marian.gunsch/02_AM_tilted_Tstep_DMI/spin-configs-99-999/spin-config-99-999-005000.dat",
            "out/03_tobi_paper/2d_plot_seebeck_110_T2_")


# %% 04

def seebeck_04(save_prefix="out/04_lowerT/seebeck_T2-2", save_prefix_eq="out/04_lowerT/seebeck_eq_T2-2"):
    print("PLOTTING SEEBECK EFFECT\n"
          "Now creating plots for SSE. Magnetization and Neel-Vector will be plotted for a temperature of 2meV."
          "Two different directions with and without DMI are plotted. ([110] and [-110])")
    print("Plotting for T=2meV")

    noDMI_kwargs = dict(alpha=0.7, linestyle="--", linewidth=1.0)

    mag_util.plot_magnetic_profile_from_paths(
        ["/data/scc/marian.gunsch/04_AM_tilted_xTstep_T2-2/spin-configs-99-999/mag-profile-99-999.altermagnet",
         "/data/scc/marian.gunsch/04_AM_tilted_yTstep_T2-2/spin-configs-99-999/mag-profile-99-999.altermagnet",
         "/data/scc/marian.gunsch/04_AM_tilted_xTstep_DMI_T2-2/spin-configs-99-999/mag-profile-99-999.altermagnet",
         "/data/scc/marian.gunsch/04_AM_tilted_yTstep_DMI_T2-2/spin-configs-99-999/mag-profile-99-999.altermagnet"], 0,
        save_prefix, None, None, 0.49, [dict(label="no DMI, [110]", **noDMI_kwargs),
                                        dict(label="no DMI, [-110]", **noDMI_kwargs),
                                        dict(label="DMI, [110]"),
                                        dict(label="DMI, [-110]"), ], "(T=2meV)")

    print("\nNow the magnetic profile with the equilibrium subtracted will be plotted.")
    prefix = "/data/scc/marian.gunsch/"
    equi_warm_noDMI = mag_util.get_mean(f"{prefix}04_AM_tilted_Tstairs_T2/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat")
    equi_warm_DMI = mag_util.get_mean(f"{prefix}04_AM_tilted_Tstairs_DMI_T2/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat")

    equi_cold_noDMI = (dict(x=0, y=0, z=1), dict(x=0, y=0, z=-1))
    equi_cold_DMI = mag_util.get_mean(f"{prefix}03_AM_tilted_Tstairs_DMI_T0/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat")

    mag_util.plot_magnetic_profile_from_paths(
        ["/data/scc/marian.gunsch/04_AM_tilted_xTstep_T2-2/spin-configs-99-999/mag-profile-99-999.altermagnet",
         "/data/scc/marian.gunsch/04_AM_tilted_yTstep_T2-2/spin-configs-99-999/mag-profile-99-999.altermagnet",
         "/data/scc/marian.gunsch/04_AM_tilted_xTstep_DMI_T2-2/spin-configs-99-999/mag-profile-99-999.altermagnet",
         "/data/scc/marian.gunsch/04_AM_tilted_yTstep_DMI_T2-2/spin-configs-99-999/mag-profile-99-999.altermagnet"], 0,
        save_prefix_eq, [equi_warm_noDMI, equi_warm_noDMI, equi_warm_DMI, equi_warm_DMI],
        [equi_cold_noDMI, equi_cold_noDMI, equi_cold_DMI, equi_cold_DMI], 0.49,
        [dict(label="no DMI, [110]", **noDMI_kwargs),
         dict(label="no DMI, [-110]", **noDMI_kwargs),
         dict(label="DMI, [110]"),
         dict(label="DMI, [-110]"), ], "(T=2meV, equilibrium subtracted)")



def spin_conservation(
        path_config_nodmi,
        path_config_dmi,
        path_bulk_nodmi,
        path_bulk_dmi,
        path_prefix="",
        title="",
        save_path=None):
    print("In past simulations, it seemed like the spin vector norm was not conserved using DMI. This is now plotted "
          "again and looked at again.")
    print(f"\t{title}\n")

    ### Spin configuration
    path_config_nodmi = path_prefix + path_config_nodmi
    path_config_dmi = path_prefix + path_config_dmi

    import os
    if os.path.exists(path_config_nodmi) or os.path.exists(path_config_dmi):
        if os.path.exists(path_config_nodmi):
            plot_2d(path_config_nodmi, title_suffix=f"(No DMI, {title})")
        if os.path.exists(path_config_dmi):
            plot_2d(path_config_dmi, title_suffix=f"(DMI, {title})")


        print("The average norms of the spin vectors* are the following: \t\t\t\t *First, the norm of every spin was "
              "taken, then these norms were averaged.")
        sel = spinconf_util.select_SL_and_component
        norm = dict()
        for info, path in zip(["No DMI", "DMI"], [path_config_nodmi, path_config_dmi]):
            if os.path.exists(path):
                conf_data = spinconf_util.read_spin_config_dat(path)
                norm[info + " A"] = np.mean(np.sqrt(sel(conf_data, "A", "x") ** 2 +
                                               sel(conf_data, "A", "y") ** 2 +
                                               sel(conf_data, "A", "z") ** 2))
                norm[info + " B"] = np.mean(np.sqrt(sel(conf_data, "B", "x") ** 2 +
                                               sel(conf_data, "B", "y") ** 2 +
                                               sel(conf_data, "B", "z") ** 2))
                print(f"{info}, SL A: \t{norm[info + " A"]:.4f}\n"
                      f"{info}, SL B: \t{norm[info + " B"]:.4f}")
        print()
    else:
        print(f"Files {path_config_nodmi} and {path_config_dmi} were not created yet!\n")


    ### Mean values
    path_bulk_nodmi = path_prefix + path_bulk_nodmi
    path_bulk_dmi = path_prefix + path_bulk_dmi
    spins_nodmi_A, spins_nodmi_B = bulk_util.get_mean(path_bulk_nodmi)
    spins_dmi_A, spins_dmi_B = bulk_util.get_mean(path_bulk_dmi)

    spins_nodmi_A['norm'] = np.sqrt(np.sum(np.array([spins_nodmi_A[c] for c in spins_nodmi_A.keys()]) ** 2))
    spins_nodmi_B['norm'] = np.sqrt(np.sum(np.array([spins_nodmi_B[c] for c in spins_nodmi_B.keys()]) ** 2))
    spins_dmi_A['norm'] = np.sqrt(np.sum(np.array([spins_dmi_A[c] for c in spins_dmi_A.keys()]) ** 2))
    spins_dmi_B['norm'] = np.sqrt(np.sum(np.array([spins_dmi_B[c] for c in spins_dmi_B.keys()]) ** 2))

    print("The norms of the averages of the spin vectors* are the following: \t\t\t\t *First averages were taken, then "
          "the norm of these averages were calculated.\n"
          f"No DMI, SL A: \t{spins_nodmi_A['norm']:.4f}\n"
          f"No DMI, SL B: \t{spins_nodmi_B['norm']:.4f}\n"
          f"DMI, SL A: \t\t{spins_dmi_A['norm']:.4f}\n"
          f"DMI, SL B: \t\t{spins_dmi_A['norm']:.4f}\n")
    print("The averages of the y and z components of the spin vectors are the following:\n"
          f"No DMI, SL A: \t Sy = {spins_nodmi_A['y']:.4f} \t and \t Sz = {spins_dmi_A['z']:.8f}\n"
          f"No DMI, SL B: \t Sy = {spins_nodmi_B['y']:.4f} \t and \t Sz = {spins_dmi_B['z']:.8f}\n"
          f"DMI, SL A: \t\t Sy = {spins_dmi_A['y']:.4f} \t and \t Sz = {spins_dmi_A['z']:.8f}\n"
          f"DMI, SL B: \t\t Sy = {spins_dmi_B['y']:.4f} \t and \t Sz = {spins_dmi_B['z']:.8f}\n")

    ## Plotting
    components = list(spins_nodmi_A.keys())    # ['x', 'y', 'z', 'norm']
    x_base = np.arange(len(components))  # [0, 1, 2, 3]

    # Offsets to separate groups
    gap = 1.5
    x_nodmi = x_base
    x_dmi = x_base + len(components) + gap  # creates visual gap

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title(title)

    # Add separation lines
    ax.axvline(x=(x_nodmi[-1] + x_dmi[0]) / 2, color='black', linewidth=1)
    ax.axhline(y=0, color='grey', linestyle='--', linewidth=1)

    # Plot points
    ax.plot(x_nodmi, [spins_nodmi_A[c] for c in components], 'o', label='no DMI A', color='blue', alpha=0.6)
    ax.plot(x_nodmi, [spins_nodmi_B[c] for c in components], '^', label='no DMI B', color='orange', alpha=0.6)
    ax.plot(x_dmi, [spins_dmi_A[c] for c in components], 'o', label='DMI A', color='green', alpha=0.6)
    ax.plot(x_dmi, [spins_dmi_B[c] for c in components], '^', label='DMI B', color='red', alpha=0.6)

    # X-ticks and labels
    ax.set_xticks(list(x_nodmi) + list(x_dmi))
    ax.set_xticklabels(components * 2)

    # Group labels
    mid_nodmi = np.mean(x_nodmi)
    mid_dmi = np.mean(x_dmi)
    ax.text(mid_nodmi, -0.05, 'no DMI', ha='center', va='top', transform=ax.get_xaxis_transform())
    ax.text(mid_dmi, -0.05, 'DMI', ha='center', va='top', transform=ax.get_xaxis_transform())

    # Other settings
    ax.set_ylabel("Mean Spin Component")
    ax.set_ylim(-1.05, 1.05)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)

    plt.show()



def check_boundaries_open_equilibrium():
    plot_2d("/data/scc/marian.gunsch/04_AM_tilted_Tstairs_DMI_T2_openbou/spin-configs-99-999/spin-config-99-999-005000.dat", width_xy=99999, save_path="out/04_lowerT/open_bound_stairs_DMI", title_suffix="(Open boundaries, DMI)")
    plot_2d("/data/scc/marian.gunsch/04_AM_tilted_Tstairs_DMI_T2/spin-configs-99-999/spin-config-99-999-005000.dat", width_xy=99999, save_path="out/04_lowerT/period_bound_stairs_DMI", title_suffix="(Periodic boundaries, DMI)")
    plot_2d("/data/scc/marian.gunsch/04_AM_tilted_Tstairs_T2_openbou//spin-configs-99-999/spin-config-99-999-005000.dat", width_xy=99999, save_path="out/04_lowerT/open_bound_stairs_noDMI", title_suffix="(Open boundaries, no DMI)")
    plot_2d("/data/scc/marian.gunsch/04_AM_tilted_Tstairs_T2/spin-configs-99-999/spin-config-99-999-005000.dat", width_xy=99999, save_path="out/04_lowerT/period_bound_stairs_noDMI", title_suffix="(Periodic boundaries, no DMI)")

    mag_util.plot_magnetic_profile_from_paths(
        ["/data/scc/marian.gunsch/04_AM_tilted_Tstairs_T2_openbou/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat",
         "/data/scc/marian.gunsch/04_AM_tilted_Tstairs_DMI_T2_openbou/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat",],
        None, "out/04_lowerT/open_bound_comparison", None, None, None,
        [dict(label="Open, no DMI", linestyle="--", alpha=0.8),
         dict(label="Open, DMI", linestyle="--", alpha=0.8),],
        dont_calculate_margins=True,
        which="yz"
    )


def presenting_data_04():
    print("[04] As previous attempts working with the equilibrium data for DMI have failed, I am trying once again "
          "with lower temperature. Turns out, the previous attempts have failed because I made a mistake with a lot "
          "of equilibrium ('stairs') simulations, mixing up two flags. I have now run the simulations again and "
          "have better (correct) results. I have correct results for the lower temperature (T=2meV). Furthermore I "
          "have also fixed the previous simulations for T=7meV. These results are therefore shown here as well."
          "Results can be seen in the folder starting with 04.")

    seebeck_03("out/04_lowerT/seebeck_T7", "out/04_lowerT/seebeck_eq_T7")
    print(seperator)

    seebeck_04()
    print(seperator)

    # return

    spin_conservation(path_config_nodmi="04_AM_tilted_Tstairs_T2/spin-configs-99-999/spin-config-99-999-005000.dat",
                      path_config_dmi="04_AM_tilted_Tstairs_DMI_T2/spin-configs-99-999/spin-config-99-999-005000.dat",
                      path_bulk_nodmi="04_AM_tilted_Tstairs_T2/04_AM_Tstairs-99-999.dat",
                      path_bulk_dmi="04_AM_tilted_Tstairs_DMI_T2/04_AM_Tstairs-99-999.dat",
                      path_prefix="/data/scc/marian.gunsch/", title="Equilibrium results for T=2meV",
                      save_path="out/04_lowerT/equilibrium_summary_T2.pdf")
    print(seperator)

    spin_conservation(path_config_nodmi="I DO NOT HAVE A 7meV EQUI CONFIG AVAILABLE",
                      path_config_dmi="/data/scc/marian.gunsch/03_AM_tilted_Tstairs_DMI/spin-configs-99-999/spin-config-99-999-005000.dat",
                      path_bulk_nodmi="data/temp/altermagnet-equilibrium-7meV.dat",
                      path_bulk_dmi="/data/scc/marian.gunsch/03_AM_tilted_Tstairs_DMI/03_AM_Tstairs-99-999.dat",
                      path_prefix="", title="Equilibrium results for T=7meV",
                      save_path="out/04_lowerT/equilibrium_summary_T7.pdf")
    print(seperator)

    spin_conservation(#path_config_nodmi="03_AM_tilted_Tstairs_DMI/spin-configs-99-999/spin-config-99-999-005000.dat",
                      path_config_nodmi="DON'T PLOT CONFIG",
                      path_config_dmi="03_AM_tilted_Tstairs_DMI/spin-configs-99-999/spin-config-99-999-005000.dat",
                      path_bulk_nodmi="03_AM_tilted_Tstairs_DMI_T0/03_AM_Tstairs_T0-99-999.dat",
                      path_bulk_dmi="03_AM_tilted_Tstairs_DMI_T0/03_AM_Tstairs_T0-99-999.dat",
                      path_prefix="/data/scc/marian.gunsch/", title="Equilibrium results for T=0meV (both sides show for DMI!!!)",
                      save_path="out/04_lowerT/equilibrium_summary_T0.pdf")
    print(seperator)

    check_boundaries_open_equilibrium()

    compare_nernst_with_equilibrium((True, True))



# %% 05 Static B field

def dispersion_relation(path_A, npy_path_A, path_B=None, title="Dispersion relation", dx=1e-10, dt=50e-16, out_path=None, rasterized=True, dont_plot=False):
    print(f"Displaying the dispersion relation of the data in path '{path_A}'...")

    mag_util.save_mag_files(path_A, npy_path_A, saving_after_index=-100000)
    data_A, data_B = mag_util.load_mag_npy_files(npy_path_A)

    skip_steps = 0

    Sx = physics.magnetization(mag_util.get_component(data_A, "x", skip_steps),
                               mag_util.get_component(data_B, "x", skip_steps))
    Sy = physics.magnetization(mag_util.get_component(data_A, "y", skip_steps),
                               mag_util.get_component(data_B, "y", skip_steps))

    Sp = Sx + 1j * Sy

    # Fourier
    time_steps = mag_util.get_component(data_A, "t", skip_steps)
    dt = time_steps[1] - time_steps[0]
    dt *= 1e-16  # unit: picoseconds TODO: unit is wrong
    dt = 50e-16

    dx = 1e-10  # 1 Angstrom    # TODO: Do not know if this is correct
    k_vectors = np.fft.fftfreq(Sp.shape[1], d=dx) # * 2 * np.pi     # TODO

    Sp_F_ = np.fft.fft2(Sp)

    freqs = np.fft.fftfreq(time_steps.shape[0], d=dt)
    freqs_shifted = np.fft.fftshift(freqs)
    omega_shifted = 2 * np.pi * freqs

    Sp_F = np.fft.fftshift(Sp_F_)
    k_shifted = np.fft.fftshift(k_vectors)

    print(f"k_min = {np.min(k_shifted)} \t k_max = {np.max(k_shifted)}")
    print(f"f_min = {np.min(freqs_shifted)} \t f_max = {np.max(freqs_shifted)}")

    magnon_density = np.abs(Sp_F) ** 2

    if dont_plot:
        return k_shifted, freqs_shifted, magnon_density

    print("Plotting...")

    # units
    k_shifted *= 1e-10  # Angstrom
    freqs_shifted *= 1e-12   # Tera

    # plotting
    fig, ax = plt.subplots()
    ax.set_title(title)

    im = ax.pcolormesh(k_shifted, freqs_shifted, magnon_density, shading='auto',
                       norm=colors.LogNorm(vmin=magnon_density.min(), vmax=magnon_density.max()),
                       rasterized=rasterized)
    # im = ax.pcolormesh(k_shifted, omega_shifted, magnon_density, shading='auto', rasterized=rasterized)

    if plot_paths:
        fig.text(0.5, 1.0, f"{path_A}", ha="center", va="top", color="green", size=6)

    # ax.set_ylabel('Frequency ω in rad/s')
    ax.set_ylabel('Frequency $f$ in THz')
    ax.set_xlabel(r'Spatial frequency $\propto k$ (1/A)')
    fig.colorbar(im, ax=ax, label='Magnon Density')

    if out_path:
        print(f"Saving dispersion relation figure to {out_path}..", end="")
        if out_path.endswith(".pdf"):
            if rasterized:
                fig.savefig(out_path, dpi=2400)
            else:
                fig.savefig(out_path)
        else:
            fig.savefig(out_path, dpi=1200)
        print(".")

    plt.show()

    print()

    return k_shifted, freqs_shifted, magnon_density


def side_by_side_comparison(dat_path_statB, npy_path_statB, dat_path_noB, npy_path_noB, direction, rasterized=True,
                            out_path=None, zoom=True, title_left="with static B", title_right="without static B"):
    k_statB, f_statB, magnon_statB = dispersion_relation(
        dat_path_statB,
        npy_path_statB,
        dont_plot=True
    )

    k_noB, f_noB, magnon_noB = dispersion_relation(
        dat_path_noB,
        npy_path_noB,
        dont_plot=True
    )

    k_statB *= 1e-10
    f_statB *= 1e-12
    k_noB *= 1e-10
    f_noB *= 1e-12

    vmin = min(magnon_statB.min(), magnon_noB.min())
    vmax = max(magnon_statB.max(), magnon_noB.max())

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True, constrained_layout=True)

    fig.suptitle(f"Dispersion relation comparison {direction}")

    im0 = axs[0].pcolormesh(k_statB, f_statB, magnon_statB, shading='auto',
                            norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                            rasterized=rasterized)
    axs[0].set_title(title_left)
    axs[0].set_xlabel(r'Spatial frequency $\propto k$ (1/A)')
    axs[0].set_ylabel('Frequency $f$ in THz')

    im1 = axs[1].pcolormesh(k_noB, f_noB, magnon_noB, shading='auto',
                            norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                            rasterized=rasterized)
    axs[1].set_title(title_right)
    axs[1].set_xlabel(r'Spatial frequency $\propto k$ (1/A)')
    axs[1].set_ylabel('Frequency $f$ in THz')

    if zoom:
        xlim = (-0.1, 0.1)
        ylim = (-25.0, 25.0)

        for ax in axs:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

    cbar = fig.colorbar(im1, ax=axs, location='right', shrink=0.9, label='Magnon Density')

    if out_path:
        print(f"Saving side by side band gap figure to {out_path}..", end="")
        if out_path.endswith(".pdf"):
            if rasterized:
                fig.savefig(out_path, dpi=2400)
            else:
                fig.savefig(out_path)
        else:
            fig.savefig(out_path, dpi=1200)
        print(".")

    plt.show()


def seebeck_05():
    print("Seebeck with static B-field")

    noB_kwargs = dict(alpha=0.7, linestyle="-.", linewidth=1.0)
    prefix = "/data/scc/marian.gunsch/"
    save_path = "out/05_staticB/default_save_file"

    prefix = "/data/scc/marian.gunsch/"
    mag_util.plot_magnetic_profile_from_paths(
        [f"{prefix}05_AM_tilted_xTstep_T2_staticB/spin-configs-99-999/mag-profile-99-999.altermagnet",
         f"{prefix}05_AM_tilted_yTstep_T2_staticB/spin-configs-99-999/mag-profile-99-999.altermagnet",
         f"{prefix}05_AM_tilted_xTstep_DMI_T2_staticB/spin-configs-99-999/mag-profile-99-999.altermagnet",
         f"{prefix}05_AM_tilted_yTstep_DMI_T2_staticB/spin-configs-99-999/mag-profile-99-999.altermagnet",
         f"{prefix}04_AM_tilted_xTstep_T2-2/spin-configs-99-999/mag-profile-99-999.altermagnet",
         f"{prefix}04_AM_tilted_yTstep_T2-2/spin-configs-99-999/mag-profile-99-999.altermagnet",
         f"{prefix}04_AM_tilted_xTstep_DMI_T2-2/spin-configs-99-999/mag-profile-99-999.altermagnet",
         f"{prefix}04_AM_tilted_yTstep_DMI_T2-2/spin-configs-99-999/mag-profile-99-999.altermagnet"],
        [0, 0, 0, 0, 0, 0, 0, 0], "out/05_staticB/seebeck_T2_completeComparison", None, None, 0.49,
        [dict(label="B=100T, no DMI, [110]"),
         dict(label="B=100T, no DMI, [-110]"),
         dict(label="B=100T, DMI, [110]"),
         dict(label="B=100T, DMI, [-110]"),
         dict(label="B=0, no DMI, [110]", **noB_kwargs),
         dict(label="B=0, no DMI, [-110]", **noB_kwargs),
         dict(label="B=0, DMI, [110]", **noB_kwargs),
         dict(label="B=0, DMI, [-110]", **noB_kwargs)], "(T=2meV)")

    mag_util.plot_magnetic_profile_from_paths(
        [f"{prefix}05_AM_tilted_xTstep_T2_staticB/spin-configs-99-999/mag-profile-99-999.altermagnet",
         f"{prefix}05_AM_tilted_yTstep_T2_staticB/spin-configs-99-999/mag-profile-99-999.altermagnet",
         f"{prefix}04_AM_tilted_xTstep_T2-2/spin-configs-99-999/mag-profile-99-999.altermagnet",
         f"{prefix}04_AM_tilted_yTstep_T2-2/spin-configs-99-999/mag-profile-99-999.altermagnet", ], [0, 0, 0, 0],
        "out/05_staticB/seebeck_T2_nodmiComparison", None, None, 0.49, [dict(label="B=100T, no DMI, [110]"),
                                                                        dict(label="B=100T, no DMI, [-110]"),
                                                                        dict(label="B=0, no DMI, [110]", **noB_kwargs),
                                                                        dict(label="B=0, no DMI, [-110]",
                                                                             **noB_kwargs)], "(T=2meV)")

    mag_util.plot_magnetic_profile_from_paths(
        [f"{prefix}05_AM_tilted_xTstep_DMI_T2_staticB/spin-configs-99-999/mag-profile-99-999.altermagnet",
         f"{prefix}05_AM_tilted_yTstep_DMI_T2_staticB/spin-configs-99-999/mag-profile-99-999.altermagnet",
         f"{prefix}04_AM_tilted_xTstep_DMI_T2-2/spin-configs-99-999/mag-profile-99-999.altermagnet",
         f"{prefix}04_AM_tilted_yTstep_DMI_T2-2/spin-configs-99-999/mag-profile-99-999.altermagnet"], [0, 0, 0, 0],
        "out/05_staticB/seebeck_T2_dmi_comparison", None, None, 0.49, [dict(label="B=100T, DMI, [110]"),
                                                                       dict(label="B=100T, DMI, [-110]"),
                                                                       dict(label="B=0, DMI, [110]", **noB_kwargs),
                                                                       dict(label="B=0, DMI, [-110]", **noB_kwargs)],
        "(T=2meV)")


def compare_with_analytical_dispersion_relation(dat_path_simulation, npy_path_simulation, mat_path_analytical,
                                                omega_selection=(0,2), title="Comparison analytical", save_path=None, plot_paths=True, rasterized=True):

    k_vals, freqs, magnon_density = dispersion_relation(
        dat_path_simulation,
        npy_path_simulation,
        dont_plot=True,
        dx=2e-10
    )

    temp = sp.io.loadmat(mat_path_analytical)
    print(temp.keys())
    omega_ana1 = temp['omega1'][omega_selection[0]]
    omega_ana2 = temp['omega1'][omega_selection[1]]
    k_ana = np.squeeze(temp['k_lin']) * 1e10 * 2

    omega_sim = freqs * 2 * np.pi
    k_vals *= 2 * np.pi

    # plotting
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_ylabel('Frequency ω in rad/s')
    ax.set_xlabel(r'k (1/m)')

    im = ax.pcolormesh(k_vals, omega_sim, magnon_density, shading='auto',
                       norm=colors.LogNorm(vmin=magnon_density.min(), vmax=magnon_density.max()),
                       rasterized=rasterized)

    if plot_paths:
        fig.text(0.5, 1.0, f"{dat_path_simulation}", ha="center", va="top", color="green", size=6)


    fig.colorbar(im, ax=ax, label='Magnon Density')

    ax.plot(k_ana[96:160], omega_ana1[96:160], color='red')
    ax.plot(k_ana[96:160], omega_ana2[96:160], color='red', label='Analytical solution')

    ax.legend()

    if save_path:
        print(f"Saving comparison to analytical to {save_path}..", end="")
        if save_path.endswith(".pdf"):
            if rasterized:
                fig.savefig(save_path, dpi=2400)
            else:
                fig.savefig(save_path)
        else:
            fig.savefig(save_path, dpi=1200)
        print(".")

    plt.show()



def presenting_data_05():
    dispersion_relation(
        "/data/scc/marian.gunsch/05_AM_tilted_Tstairs_T2_x_Bstatic/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat",
        "data/05_staticB/stairs_T2_x_staticB_A.npy",
        title="Dispersion relation [110] (with static B-field)",
        # out_path="out/05_staticB/dispersionRel_statB_110.png"
        # out_path="out/05_staticB/dispersionRel_statB_110.pdf"
    )

    dispersion_relation(
        "/data/scc/marian.gunsch/05_AM_tilted_Tstairs_T2_y_Bstatic/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat",
        "data/05_staticB/stairs_T2_y_staticB_A.npy",
        title="Dispersion relation [-110] (with static B-field)",
        # out_path="out/05_staticB/dispersionRel_statB_-110.png"
        # out_path="out/05_staticB/dispersionRel_statB_-110.pdf"
    )

    dispersion_relation(
        "/data/scc/marian.gunsch/05_AM_tilted_Tstairs_T2_x/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat",
        "data/05_staticB/stairs_T2_x_A.npy",
        title="Dispersion relation [110] (no static B field, no DMI)",
        # out_path="out/05_staticB/dispersionRel_110.png"
        # out_path="out/05_staticB/dispersionRel_110.pdf"
    )

    dispersion_relation(
        "/data/scc/marian.gunsch/05_AM_tilted_Tstairs_T2_y/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat",
        "data/05_staticB/stairs_T2_y_A.npy",
        title="Dispersion relation [-110] (no static B field, no DMI)",
        # out_path="out/05_staticB/dispersionRel_-110.png"
        # out_path="out/05_staticB/dispersionRel_-110.pdf"
    )

    side_by_side_comparison(
        "/data/scc/marian.gunsch/05_AM_tilted_Tstairs_T2_x_Bstatic/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat",
                            "data/05_staticB/stairs_T2_x_staticB_A.npy",
                            "/data/scc/marian.gunsch/05_AM_tilted_Tstairs_T2_x/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat",
                            "data/05_staticB/stairs_T2_x_A.npy",
                            "[110]",
                            out_path="out/05_staticB/band_gap_comparison_110.pdf"
                            )
    side_by_side_comparison(
        "/data/scc/marian.gunsch/05_AM_tilted_Tstairs_T2_y_Bstatic/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat",
                            "data/05_staticB/stairs_T2_y_staticB_A.npy",
                            "/data/scc/marian.gunsch/05_AM_tilted_Tstairs_T2_y/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat",
                            "data/05_staticB/stairs_T2_y_A.npy",
                            "[-110]",
                            out_path="out/05_staticB/band_gap_comparison_-110.pdf"
                            )

    side_by_side_comparison(
        "/data/scc/marian.gunsch/05_AM_tilted_Tstairs_T2_x_Bstatic/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat",
        "data/05_staticB/stairs_T2_x_staticB_A.npy",
        "/data/scc/marian.gunsch/05_AM_tilted_Tstairs_T2_x/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat",
        "data/05_staticB/stairs_T2_x_A.npy",
        "[110]",
        out_path="out/05_staticB/dispersionRel_comparison_110.pdf",
        zoom=False
        )
    side_by_side_comparison(
        "/data/scc/marian.gunsch/05_AM_tilted_Tstairs_T2_y_Bstatic/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat",
        "data/05_staticB/stairs_T2_y_staticB_A.npy",
        "/data/scc/marian.gunsch/05_AM_tilted_Tstairs_T2_y/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat",
        "data/05_staticB/stairs_T2_y_A.npy",
        "[-110]",
        out_path="out/05_staticB/dispersionRel_comparison_-110.pdf",
        zoom=False
        )

    seebeck_05()

    compare_with_analytical_dispersion_relation(
        "/data/scc/marian.gunsch/05_AM_tilted_Tstairs_T2_x_Bstatic/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat",
        "data/05_staticB/stairs_T2_x_staticB_A.npy",
        "data/05_staticB/noDMI_B.mat",
        title="B=100T, no DMI, sim in [110]",
        save_path="out/05_staticB/dispersion_ana_110_B100_nodmi.png",
    )

    compare_with_analytical_dispersion_relation(
        "/data/scc/marian.gunsch/05_AM_tilted_Tstairs_T2_y_Bstatic/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat",
        "data/05_staticB/stairs_T2_y_staticB_A.npy",
        "data/05_staticB/noDMI_B.mat",
        title="B=100T, no DMI, sim in [-110]",
        save_path="out/05_staticB/dispersion_ana_n110_B100_nodmi.png",
        omega_selection=(1,3)
    )

    compare_with_analytical_dispersion_relation(
        "/data/scc/marian.gunsch/05_AM_tilted_Tstairs_T2_x/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat",
        "data/05_staticB/stairs_T2_x_A.npy",
        "data/05_staticB/noDMI_noB.mat",
        title="B=0, no DMI, sim in [110]",
        save_path="out/05_staticB/dispersion_ana_110_B0_nodmi.png",
    )

    compare_with_analytical_dispersion_relation(
        "/data/scc/marian.gunsch/05_AM_tilted_Tstairs_T2_y/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat",
        "data/05_staticB/stairs_T2_y_A.npy",
        "data/05_staticB/noDMI_noB.mat",
        title="B=0, no DMI, sim in [-110]",
        save_path="out/05_staticB/dispersion_ana_n110_B0_nodmi.png",
        omega_selection=(1,3)
    )

    compare_with_analytical_dispersion_relation(
        "/data/scc/marian.gunsch/06_AM_tilted_Tstairs_T2_x_Bn100/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat",
        "data/06_staticB/stairs_T2_x_Bn100_A.npy",
        "data/05_staticB/noDMI_B-100T.mat",
        title="B=-100T, no DMI, sim in [110]",
        save_path="out/05_staticB/dispersion_ana_110_Bn100_nodmi.png",
    )

    compare_with_analytical_dispersion_relation(
        "/data/scc/marian.gunsch/06_AM_tilted_Tstairs_T2_y_Bn100/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat",
        "data/06_staticB/stairs_T2_y_Bn100_A.npy",
        "data/05_staticB/noDMI_B-100T.mat",
        title="B=-100T, no DMI, sim in [-110]",
        save_path="out/05_staticB/dispersion_ana_n110_Bn100_nodmi.png",
        omega_selection=(1,3)
    )


# %% 06 Static B field: effect of negative field, field dependence of SSE, non zero equilibrium



def negative_B_equi():
    print("EFFECT OF A NEGATIVE B FIELD ON EQUILIBRIUM")
    path_110 = "/data/scc/marian.gunsch/06_AM_tilted_Tstairs_T2_x_Bn100/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat"
    path_n110 = "/data/scc/marian.gunsch/06_AM_tilted_Tstairs_T2_y_Bn100/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat"

    npy_path_110 = "data/06_staticB/stairs_T2_x_Bn100_A.npy"
    npy_path_n110 = "data/06_staticB/stairs_T2_y_Bn100_A.npy"
    dispersion_relation(
        path_110,
        npy_path_110,
        title="Dispersion relation [110] (with B=-100T)",
        out_path="out/06_staticB/dispersionRel_Bn100_-110.pdf"
    )

    dispersion_relation(
        path_n110,
        npy_path_n110,
        title="Dispersion relation [-110] (with B=-100T)",
        out_path="out/06_staticB/dispersionRel_Bn100_-110.pdf"
    )

    side_by_side_comparison(
        path_110,
        npy_path_110,
        "/data/scc/marian.gunsch/05_AM_tilted_Tstairs_T2_x/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat",
        "data/05_staticB/stairs_T2_x_A.npy",
        "[110]",
        title_left="B=-100T",
        out_path= "out/06_staticB/dispersionRel_Bn100B0_110.pdf",
        zoom=False
        )
    side_by_side_comparison(
        path_n110,
        npy_path_n110,
        "/data/scc/marian.gunsch/05_AM_tilted_Tstairs_T2_y/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat",
        "data/05_staticB/stairs_T2_y_A.npy",
        "[-110]",
        title_left="B=-100T",
        out_path="out/06_staticB/dispersionRel_Bn100B0_-110.pdf",
        zoom=False
        )


    side_by_side_comparison(
        "/data/scc/marian.gunsch/05_AM_tilted_Tstairs_T2_x_Bstatic/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat",
        "data/05_staticB/stairs_T2_x_staticB_A.npy",
        path_110,
        npy_path_110,
        "[110]",
        title_left="B=+100T",
        title_right="B=-100T",
        out_path="out/06_staticB/dispersionRel_Bn100B100_110.pdf",
        zoom=False
        )
    side_by_side_comparison(
        "/data/scc/marian.gunsch/05_AM_tilted_Tstairs_T2_y_Bstatic/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat",
        "data/05_staticB/stairs_T2_y_staticB_A.npy",
        path_n110,
        npy_path_n110,
        "[-110]",
        title_left="B=+100T",
        title_right="B=-100T",
        out_path="out/06_staticB/dispersionRel_Bn100B100_-110.pdf",
        zoom=False
        )



def negative_B_SSE():
    print("EFFECT OF A NEGATIVE B FIELD ON SPIN SEEBECK EFFECT")
    path_110 = "/data/scc/marian.gunsch/06_AM_tilted_xTstep_T2_Bn100/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat"
    path_n110 = "/data/scc/marian.gunsch/06_AM_tilted_yTstep_T2_Bn100/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat"

    prefix = "/data/scc/marian.gunsch/"

    noB_kwargs = dict(alpha=0.7, linestyle="-.", linewidth=1.0)
    posB_kwargs = dict(linestyle="--")

    kwargs_list = [dict(label="B=-100T, [110]"),
         dict(label="B=-100T, [-110]"),
         dict(label="B=+100T, [110]",
              **posB_kwargs),
         dict(label="B=+100T, [-110]",
              **posB_kwargs),
         dict(label="B=0, [110]", **noB_kwargs),
         dict(label="B=0, [-110]", **noB_kwargs)]

    _, _, magn_list, _ = mag_util.plot_magnetic_profile_from_paths(
        [path_110,
           path_n110,
           f"{prefix}05_AM_tilted_xTstep_T2_staticB/spin-configs-99-999/mag-profile-99-999.altermagnet",
           f"{prefix}05_AM_tilted_yTstep_T2_staticB/spin-configs-99-999/mag-profile-99-999.altermagnet",
           f"{prefix}04_AM_tilted_xTstep_T2-2/spin-configs-99-999/mag-profile-99-999.altermagnet",
           f"{prefix}04_AM_tilted_yTstep_T2-2/spin-configs-99-999/mag-profile-99-999.altermagnet", ],
          None,
        "out/06_staticB/seebeck_nB",
        None, None, None,
        kwargs_list,
        which = "z")

    # Analyzing the peaks
    lower, upper = 115, 135

    for magnetization, kwargs in zip(magn_list, kwargs_list):
        for component in magnetization.keys():
            label = kwargs["label"]
            data = magnetization[component][lower:upper]

            min_ = np.min(data)
            max_ = np.max(data)
            diff_ = max_ - min_
            print(f"{label}, {component}: \t local max: {max_:.6f} \t local min: {min_:.6f} \t difference: {diff_:.6f}")

    observation = (
    "For B=0 one can see \n" 
        "\t that the difference for [110] is smaller than for [-110] \n"
        "\t That the maximum for [110] is 0.000003 greater than for [-110] \n"
        "\t That the (abs val of) minimum for [110] is 0.00002 smaller than for [-110] \n"
    "Comparing B=+-100T one can see \n"
        "\t B=+100T decreases difference for [110], increases difference for [-110] \n"
        "\t B=-100T increases difference for [110], decreases difference for [-110] \n"
        "\t The smaller difference in cases B=+-100T is smaller for [110] than for [-110], for B=0T the difference is also smaller for [110] \n"
        "\t The bigger difference in cases B=+-100T is smaller for [110] than for [-110], for B=0T the difference is also smaller for [110] \n")

    print(observation)


def non_zero_Teq_SSE_statB():
    print("EFFECT OF A FINITE TEMPERATURE ON THE RIGHT WHEN HAVING A STATIC MAGNETIC FIELD")
    path_110 = "/data/scc/marian.gunsch/06_AM_tilted_xTstep_T2Teq05_B100/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat"
    path_n110 = "/data/scc/marian.gunsch/06_AM_tilted_yTstep_T2Teq05_B100/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat"

    path_110 = "/data/scc/marian.gunsch/06_AM_tilted_xTstep_T15Teq05_B100/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat"
    path_n110 = "/data/scc/marian.gunsch/06_AM_tilted_yTstep_T15Teq05_B100/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat"

    mag_util.plot_magnetic_profile_from_paths(
        [path_110,
         path_n110,
         "/data/scc/marian.gunsch/05_AM_tilted_xTstep_T2_staticB/spin-configs-99-999/mag-profile-99-999.altermagnet",
         "/data/scc/marian.gunsch/05_AM_tilted_yTstep_T2_staticB/spin-configs-99-999/mag-profile-99-999.altermagnet"],
        None,
        "out/06_staticB/seebeck_B100_Teq05",
        None,
        None,
        None,
        [dict(label="B=100T, k*Teq=0.5meV, [110]"),
         dict(label="B=100T, k*Teq=0.5meV, [-110]"),
         dict(label="B=100T, Teq=0, [110]", alpha=0.8, linewidth=0.7, linestyle="--"),
         dict(label="B=100T, Teq=0, [-110]", alpha=0.8, linewidth=0.7, linestyle="--")],
         which="z"
    )


def get_path_from_direction_field_strength(direction: str, field_strength: int):
    simulation_index = "05"
    if field_strength == 0:
        return f"/data/scc/marian.gunsch/04_AM_tilted_{direction}Tstep_T2-2/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat"
    if field_strength < 0:
        field_strength = f"n{int(np.abs(field_strength))}"
        simulation_index = "06"
    return (f"/data/scc/marian.gunsch/{simulation_index}_AM_tilted_{direction}Tstep_T2_B{field_strength}"
            f"/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat")


def propagation_length_SSE_depend_on_B(field_strength_list=None, suppress_plots=[True, True, False], fit_plot_save_path=None,
                                       alpha_plot_path=None, prop_length_plot_path=None, other_fit_func=True):
    print("FITS AND PROPAGATION LENGTH")

    field_strength_list = field_strength_list or [-100, 0, 50, 60, 70, 80, 90, 100]
    direction_list = ["x", "y"]
    direction_name_list = ["[110]", "[-110]"]
    color_list = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray"]

    x_bot = 127     # or 126
    x_top = 200

    alpha_list_dict = dict(x=[], y=[])
    beta_list_dict = dict(x=[], y=[])

    fig, ax = plt.subplots()
    ax.set_title(f"B dependent Seebeck with Fits")

    for field_strength, color in zip(field_strength_list, color_list):
        for direction, cryst_direction in zip(direction_list, direction_name_list):
            print(f"Calculating fit for B={field_strength}T, {cryst_direction}")

            path = get_path_from_direction_field_strength(direction, field_strength)
            data_A = np.loadtxt(path)
            data_B = np.loadtxt(mag_util.infer_path_B(path))
            spin_A, spin_B = mag_util.get_component(data_A, which="z"), mag_util.get_component(data_B, which="z")
            magnetization = physics.magnetization(spin_A, spin_B, True)

            x_data = np.arange(len(magnetization), dtype=float)

            local_extremum = np.argmax(np.abs(magnetization)[x_bot:x_top]) + x_bot + 1  # TODO: maybe we want to be one to the right of the actual extremum

            data_for_fit = magnetization[local_extremum:x_top]

            x_data_for_fit = np.arange(len(data_for_fit), dtype=float)  # TODO - need to know distance between lattice points

            fit_func = util.exp_fit_func_2 if other_fit_func else util.exp_fit_func

            if not other_fit_func:
                initial_A = 0.001 if direction == "x" else -0.0007
                popt, pcov = curve_fit(fit_func, x_data_for_fit, data_for_fit, p0=[initial_A, 0.1])
            else:
                initial_A = 0.0005 if direction == "x" else -0.0004
                initial_B = 0.0005 if direction == "x" else -0.0004
                popt, pcov = curve_fit(fit_func, x_data_for_fit, data_for_fit,
                                       p0=[initial_A, 0.1, initial_B, 0.1], maxfev=10000)

            print(f"direction={direction}, B={field_strength}T, popt={popt}", )

            alpha_list_dict[direction].append(popt[1])
            if other_fit_func:
                beta_list_dict[direction].append(popt[3])

            if suppress_plots[0]:
                continue
            x_plot_fit = x_data - local_extremum

            ax.plot(x_data, magnetization, label=f"B={field_strength}T, {cryst_direction}",
                    linestyle="-", linewidth=0.3, marker="o", markersize=0.8, alpha=0.8, color=color)
            ax.plot(x_data, fit_func(x_plot_fit, *popt), label=f"Fit: B={field_strength}T, {cryst_direction}",
                    linestyle="--", linewidth=0.6, alpha=0.7, color=color[4:])


    if not suppress_plots[0]:
        ax.set_ylim(-0.0018, 0.0018)

        ax.legend(fontsize='xx-small', title_fontsize="xx-small", ncols=2)

        if fit_plot_save_path:
            print(f"Saving to {fit_plot_save_path}")
            fig.savefig(fit_plot_save_path)

        fig.show()

    np.save("data/06_staticB/alpha_110.npy", alpha_list_dict['x'])
    np.save("data/06_staticB/alpha_n110.npy", alpha_list_dict['y'])

    if not suppress_plots[1]:
        fig, ax = plt.subplots()
        ax.set_title("alpha dependent on B")
        ax.set_xlabel("B field in T")
        ax.set_ylabel("alpha in unit 1/index (no dimension)")
        ax.plot(field_strength_list, alpha_list_dict['x'], label="alpha [110]", marker="o", linestyle="")
        ax.plot(field_strength_list, alpha_list_dict['y'], label="alpha [-110]", marker="s", linestyle="")

        if other_fit_func:
            ax.plot(field_strength_list, beta_list_dict['x'], label="beta [110]", marker="o", linestyle="")
            ax.plot(field_strength_list, beta_list_dict['y'], label="beta [-110]", marker="s", linestyle="")

        ax.legend()

        if alpha_plot_path:
            print(f"Saving to {alpha_plot_path}")
            fig.savefig(alpha_plot_path)

        fig.show()

        fig, ax = plt.subplots()
        ax.set_title("propagation length dependent on B")
        ax.set_xlabel("B field in T")
        ax.set_ylabel("propagation length in unit index (no dimension)")
        ax.plot(field_strength_list, 1.0 / np.array(alpha_list_dict['x']), label="1/alpha [110]", marker="o", linestyle="")
        ax.plot(field_strength_list, 1.0 / np.array(alpha_list_dict['y']), label="1/alpha [-110]", marker="o", linestyle="")

        if other_fit_func:
            ax.plot(field_strength_list, 1.0 / np.array(beta_list_dict['x']), label="1/beta [110]", marker="o", linestyle="")
            ax.plot(field_strength_list, 1.0 / np.array(beta_list_dict['y']), label="1/beta [-110]", marker="s", linestyle="")

        ax.legend()

        if prop_length_plot_path:
            print(f"Saving to {prop_length_plot_path}")
            fig.savefig(prop_length_plot_path)

        fig.show()

def magnon_acc_SSE_depend_on_B(field_strength_list=None, seebeck_plot_path=None, seebeck_magnon_acc_path=None):
    print("MAGNETIZATION AND MAGNON ACCUMULATION")

    field_strength_list = field_strength_list or [-100, 0, 50, 60, 70, 80, 90, 100]
    direction_list = ["x", "y"]
    direction_name_list = ["[110]", "[-110]"]
    color_list = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray"]
    linestyle_list = ["-", "-."]

    path_list = []
    kwargs_list = []

    for field_strength, color in zip(field_strength_list, color_list):
        for direction, cryst_direction, linestyle in zip(direction_list, direction_name_list, linestyle_list):
            path_list.append(get_path_from_direction_field_strength(direction, field_strength))
            kwargs_list.append(dict(label=f"B={field_strength}T, {cryst_direction}", color=color, linestyle=linestyle,
                                    linewidth=0.6, alpha=0.8))

    spins_A_list, spins_B_list = mag_util.load_from_path_list(path_list, which="z", return_raw_list=False)

    mag_util.plot_magnetic_profile(
        spins_A_list, spins_B_list, seebeck_plot_path, None, None,
        None, kwargs_list)

    # Now with subtracting equilibrium
    lower_index = 10
    upper_index = 50

    spins_A_list_equi = [dict(z=spins_A['z'][lower_index:upper_index]) for spins_A in spins_A_list]
    spins_B_list_equi = [dict(z=spins_B['z'][lower_index:upper_index]) for spins_B in spins_B_list]

    mag_util.plot_magnetic_profile(
        spins_A_list_equi, spins_B_list_equi, None, None, None,
        None, kwargs_list)

    spins_equi_values = [(dict(z=np.mean(spins_A['z'])), dict(z=np.mean(spins_B['z'])))
                         for spins_A, spins_B in zip(spins_A_list_equi, spins_B_list_equi)]

    mag_util.plot_magnetic_profile(
        spins_A_list, spins_B_list, seebeck_magnon_acc_path, spins_equi_values, None,
        None, kwargs_list)


def presenting_data_06():
    print("[06] To further investigate the effect of a static magnetic field (in z direction) on an altermagnet, the "
          "following things are analyzed:\n"
          "- The effect of a negative B field in equilibrium. Do the bands shift in the opposite way? (As we would "
          "expect)\n"
          "- The effect of a negative B field on the SSE. Therefore a simulation with a temperature step is "
          "analyzed.\n"
          "- Spin Seebeck Effect with a static B field and a nonzero equilibrium temperature on the right.\n"
          "- The effect of different field strengths on the SSE: 50 to 100 Tesla with 10 Tesla steps:\n"
          "\t - The different maxima for different crystallographic directions\n"
          "\t - The different propagation lengths for different crystallographic directions and different B fields.\n"
          "\t - The propagation lengths plotted as a function of B-field (for both directions), fit for value where "
          "propagation length for both directions are equal.\n")

    print(seperator)

    negative_B_equi()

    print(seperator)

    negative_B_SSE()

    print(seperator)

    non_zero_Teq_SSE_statB()

    print(seperator)
    print("ANALYSIS OF SPIN SEEBECK EFFECT FOR DIFFERENT MAGNETIC FIELD STRENGTHS")

    propagation_length_SSE_depend_on_B(None, [False, False],
                                       "out/06_staticB/sse_Bdepend_fits.pdf", "out/06_staticB/alpha_dep_on_B.pdf",
                                       "out/06_staticB/prop_length_dep_on_B.pdf", False)
    # magnetic_field_dependence_SSE(None, [False, False],
    #                               "out/06_staticB/seebeck_mag_profile_B50B100-2",
    #                               "out/06_staticB/sse_Bdepend_fits-2.pdf",
    #                               "out/06_staticB/alpha_dep_on_B-2.pdf",
    #                               "out/06_staticB/prop_length_dep_on_B-2.pdf",
    #                               True)

    magnon_acc_SSE_depend_on_B(None, "out/06_staticB/seebeck_mag_profile_B50B100",
                               "out/06_staticB/seebeck_delta_mag_neel_B50B100")

    print(seperator)


# %% 02,04,07 Verifying Spin Nernst (1/?)



# TODO: Have not checked if the thing with step direction works

def spin_current_nernst(config_path, step_direction='x', is_tilted=True, distance_from_step=70, save_suffix="", title_suffix=""):
    config_file = config_path

    config_data = spinconf_util.read_spin_config_dat(config_file, is_tilted, fixed_version=True)

    direction = "transversal"

    *spin_currents, j5 = spinconf_util.average_z_layers(*spinconf_util.calculate_spin_currents(config_data, direction, True))
    spin_currents = [np.squeeze(spin_current) for spin_current in spin_currents]

    N = 256
    rel_T_pos = 0.49
    T_step = helper.get_absolute_T_step_index(rel_T_pos, N)

    for left, right, title, save_path in zip(
            [T_step - 5, 0, T_step + distance_from_step, 0],
            [T_step + 5, T_step - distance_from_step, N, -1],
            [f"near step ({direction}) {title_suffix}", f"warm region ({direction}) {title_suffix}",
             f"cold region ({direction}) {title_suffix}", f"complete grid ({direction}) {title_suffix}"],
            [f"out/08_nernst/nearStep_{save_suffix}", f"out/08_nernst/warmRegion_{distance_from_step}_{save_suffix}",
             f"out/08_nernst/coldRegion_{distance_from_step}_{save_suffix}", f"out/08_nernst/completeRegion_{save_suffix}"]
    ):

        if step_direction == 'x':
            spin_currents_cutout = [spin_current[left:right] for spin_current in spin_currents]
            axes = (1,)       # average over x axis, and single z layer
        elif step_direction == 'y':
            spin_currents_cutout = [spin_current[:, left:right] for spin_current in spin_currents]
            axes = (0,)
        else:
            raise ValueError("step_direction must be 'x' or 'y'")

        X, Y = np.meshgrid(np.arange(0, spin_currents_cutout[0].shape[0], 1, dtype=int),
                           np.arange(0, spin_currents_cutout[0].shape[1], 1, dtype=int),
                           sparse=True, indexing="xy")

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(title)

        axs = axs.flatten()
        titles = ["j_inter_p", "j_inter_m", "j_intra_A", "j_intra_B"]

        for i in range(len(titles)):
            titles[i] = f"{titles[i]}, transversal"

        for ax, data_grid, title_ in zip(axs, spin_currents_cutout, titles):
            im = ax.pcolormesh(X, Y, data_grid.T, norm=colors.CenteredNorm(), cmap='RdBu_r')
            ax.set_title(title_)
            ax.set_aspect('equal', 'box')
            ax.margins(x=0, y=0)
            fig.colorbar(im, ax=ax)

        fig.tight_layout()

        if True:
            fig.text(0.5, 0.5, config_file, color="green", ha="center", va="center")

        if save_path:
            save_path_ = f"{save_path}.pdf"
            print(f"Saving to {save_path_}...")
            fig.savefig(save_path_)

        plt.show()


        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(title)

        axs = axs.flatten()
        titles = ["j_inter_p", "j_inter_m", "j_intra_A", "j_intra_B"]

        for ax, data_grid, title_ in zip(axs, spin_currents_cutout, titles):
            spin_current_profile = np.mean(data_grid, axis=axes)
            ax.set_title(title_)
            ax.set_xlabel("Index (position)")
            ax.set_ylabel("spin current")
            x_axis = np.arange(spin_current_profile.shape[0]) + left
            ax.plot(x_axis, spin_current_profile)

        if save_path:
            fig.savefig(f"{save_path}_profile.pdf")
        plt.show()





def presenting_data_07():
    # spin_current_nernst("/data/scc/marian.gunsch/07_AM_tilted_xTstep_y/spin-configs-99-999/spin-config-99-999-005000.dat")
    spin_current_nernst("/data/scc/marian.gunsch/07_AM_tilted_xTstep_yABC_/spin-configs-99-999/spin-config-99-999-005000.dat",
                        save_suffix="110_", title_suffix="110_")
    spin_current_nernst("/data/scc/marian.gunsch/07_AM_tilted_xTstep_yABC_/spin-configs-99-999/spin-config-99-999-005000.dat",
                        distance_from_step=30, save_suffix="110_", title_suffix="110_")

# %% 08 Magnetization Nernst / Verifying Spin Nernst (2/?)


def compare_nernst_with_equilibrium(run=(True, True)):
    # equiT2_open_path = "/data/scc/marian.gunsch/04_AM_tilted_Tstairs_T2_openbou/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat"
    equiT2_open_path = "/data/scc/marian.gunsch/04_AM_tilted_yTstairs_T2_openbou/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat"
    # No ABC, open, open, open

    # equiT2DMI_open_path = "/data/scc/marian.gunsch/04_AM_tilted_Tstairs_DMI_T2_openbou/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat"
    equiT2DMI_open_path = "/data/scc/marian.gunsch/04_AM_tilted_yTstairs_DMI_T2_openbou/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat"

    nernstT4_path = "/data/scc/marian.gunsch/AM_tiltedX_Tstep_nernst_T4/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat"
    # No ABC, open open open

    equiT2_path_bulk = "/data/scc/marian.gunsch/AM_tiltedX_ttmstairs_T2meV/AM_Teq2meV-99-999.dat"
    equiT7_path_bulk = "data/temp/altermagnet-equilibrium-7meV.dat"
    nernstT2_path = "/data/scc/marian.gunsch/AM_tiltedX_Tstep_nernst_T2/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat"
    nernstT7_path = "/data/scc/marian.gunsch/AM_tiltedX_Tstep_nernst_T7/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat"

    def compare_nernst_with_equilibrium_1():
        mag_util.plot_magnetic_profile_from_paths(
            [equiT2_open_path, nernstT4_path], None, None, None, None, None,
            [dict(label="T=2meV, open"), dict(label="nernst effect, Delta T = 4 meV")],
            which="z", dont_calculate_margins=True
        )

        # Checking where the magnetization comes from
        step_pos = 0.49
        N = 256
        warm_portion = helper.get_absolute_T_step_index(step_pos, N) / N
        cold_portion = 1.0 - warm_portion

        for equi_path, sne_path, T in zip([equiT2_path_bulk, equiT7_path_bulk], [nernstT2_path, nernstT7_path], [2, 7]):
            print(f"Comparison for T={T}meV")
            temp_A, temp_B = bulk_util.get_mean(equi_path)
            eq_spin_z_A = temp_A['z']
            eq_spin_z_B = temp_B['z']

            spinT0_z_A = 1
            spinT0_z_B = -1

            eq_avg_spin_A = warm_portion * eq_spin_z_A + cold_portion * spinT0_z_A  # average spin if we clipped two equilibriums
            eq_avg_spin_B = warm_portion * eq_spin_z_B + cold_portion * spinT0_z_B  # together (left Twarm, right T0)

            sne_spin_z_A = mag_util.time_avg(mag_util.get_component(np.loadtxt(sne_path)))
            sne_spin_z_B = mag_util.time_avg(mag_util.get_component(np.loadtxt(mag_util.infer_path_B(sne_path))))

            sne_avg_spin_A = np.mean(sne_spin_z_A[15:-15])
            sne_avg_spin_B = np.mean(sne_spin_z_B[15:-15])

            print(f"eq_avg_spin_A =  {eq_avg_spin_A:.5f} \t eq_avg_spin_B =  {eq_avg_spin_B:.5f}")
            print(f"sne_avg_spin_A = {sne_avg_spin_A:.5f} \t sne_avg_spin_B = {sne_avg_spin_B:.5f}")

            eq_magn = physics.magnetization(eq_avg_spin_A, eq_avg_spin_B)
            eq_neel = physics.neel_vector(eq_avg_spin_A, eq_avg_spin_B)
            sne_magn = physics.magnetization(sne_avg_spin_A, sne_avg_spin_B)
            sne_neel = physics.neel_vector(sne_avg_spin_A, sne_avg_spin_B)

            print(f"eq_magn =  {eq_magn:.5f} \t eq_neel =  {eq_neel:.5f}")
            print(f"sne_magn = {sne_magn:.5f} \t sne_neel = {sne_neel:.5f}")
            print()
        print("What I have seen now, is that there is some sort of effect, could also be SSE tho.")


    def compare_nernst_with_equilibrium_2():
        print("2")
        sne_spin_z_A = mag_util.time_avg(mag_util.get_component(np.loadtxt(nernstT4_path)))   # T4 or T2?
        sne_spin_z_B = mag_util.time_avg(mag_util.get_component(np.loadtxt(mag_util.infer_path_B(nernstT4_path))))

        eqopen_spin_z_A = mag_util.time_avg(mag_util.get_component(np.loadtxt(equiT2_open_path)))[::1]   # TODO [::-1] or not,
        eqopen_spin_z_B = mag_util.time_avg(mag_util.get_component(np.loadtxt(mag_util.infer_path_B(equiT2_open_path))))[::1]

        eqopen_magn = physics.magnetization(eqopen_spin_z_A, eqopen_spin_z_B)
        sne_magn = physics.magnetization(sne_spin_z_A, sne_spin_z_B)

        if True is True:
            fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 4))

            fig.suptitle("")

            ax1.plot(sne_spin_z_A, label="SNE (T=4meV), SL A, [-110]", marker="o", linestyle="--")
            ax2.plot(sne_spin_z_A, marker="o", linestyle="--")
            ax1.plot(np.abs(sne_spin_z_B), label="SNE (T=4meV), SL B, [-110]", marker="o", linestyle="--")
            ax2.plot(np.abs(sne_spin_z_B), marker="o", linestyle="--")
            ax1.plot(eqopen_spin_z_A, label="equi (T=2meV), SL A, [-110]", marker="s", linestyle="--")
            ax2.plot(eqopen_spin_z_A, marker="s", linestyle="--")
            ax1.plot(np.abs(eqopen_spin_z_B), label="equi (T=2meV), SL B, [-110]", marker="s", linestyle="--")
            ax2.plot(np.abs(eqopen_spin_z_B), marker="s", linestyle="--")

            ax1.set_xlim(-0.5, 6.2)
            ax2.set_xlim(248.8, 255.5)

            ax1.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax1.yaxis.tick_left()
            ax2.yaxis.tick_right()

            ax1.legend()

            fig.savefig("out/nernst_comparison_equi-3.pdf")

            plt.tight_layout()
            plt.show()
        if True is True:
            fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 4))

            fig.suptitle("Magnetization")

            ax1.plot(eqopen_magn, label="Equi (T=2meV), [-110]", marker="s", linestyle="--")
            ax2.plot(eqopen_magn, marker="s", linestyle="--")
            ax1.plot(sne_magn, label="SNE (T=4meV), [-110]", marker="o", linestyle="--")
            ax2.plot(sne_magn, marker="o", linestyle="--")

            ax1.set_xlim(-0.5, 6.2)
            ax2.set_xlim(248.8, 255.5)

            ax1.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax1.yaxis.tick_left()
            ax2.yaxis.tick_right()

            ax1.legend()

            fig.savefig("out/nernst_comparison_equi-magn.pdf")

            plt.tight_layout()
            plt.show()



    if run[0]:
        compare_nernst_with_equilibrium_1()
    if run[1]:
        compare_nernst_with_equilibrium_2()


def nernst_manual_layer_averaging(config_path, step_direction='x', is_tilted=True, distance_from_step=70, save_suffix="", title_suffix=""):
    config_data = spinconf_util.read_spin_config_dat(config_path, is_tilted=is_tilted)

    N = 256
    rel_T_pos = 0.49
    T_step = helper.get_absolute_T_step_index(rel_T_pos, N)

    for left, right, title, save_path in zip(
            [T_step - 5, 0, T_step + distance_from_step],
            [T_step + 5, T_step - distance_from_step, N],
            [f"near step {title_suffix}", f"warm region {title_suffix}", f"cold region {title_suffix}"],
            [f"out/08_nernst/nearStep_{save_suffix}", f"out/08_nernst/warmRegion_{distance_from_step}_{save_suffix}",
             f"out/08_nernst/coldRegion_{distance_from_step}_{save_suffix}"]
    ):
        if step_direction == 'x':
            cut_data = spinconf_util.average_z_layers(config_data[:, left:right])
            axes = (1, 2)       # average over x axis, and single z layer
        elif step_direction == 'y':
            cut_data = spinconf_util.average_z_layers(config_data[left:right])
            axes = (0, 2)
        else:
            raise ValueError("step_direction must be 'x' or 'y'")

        magnetization_config = physics.magnetization(spinconf_util.select_SL_and_component(cut_data, 'A', 'z'),
                                                     spinconf_util.select_SL_and_component(cut_data, 'B', 'z'))

        magnetization_profile = np.average(magnetization_config, axis=axes)


        data_grid = np.squeeze(magnetization_config)
        X, Y = np.meshgrid(np.arange(left, right, 1, dtype=int),
                           np.arange(0, data_grid.shape[0], 1, dtype=int),
                           sparse=True, indexing='xy')

        fig, ax = plt.subplots()
        ax.set_aspect('equal', 'box')
        ax.set_title(title)
        im = ax.pcolormesh(X, Y, data_grid, norm=colors.CenteredNorm(), cmap='RdBu_r')
        fig.colorbar(im, ax=ax)
        ax.margins(x=0, y=0)
        fig.tight_layout()
        fig.savefig(f"{save_path}_config.pdf")
        plt.show()

        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel("Index (position)")
        ax.set_ylabel("magnetization")
        ax.plot(magnetization_profile)
        fig.savefig(f"{save_path}_profile.pdf")
        plt.show()


def presenting_data_08():
    # compare_nernst_with_equilibrium((False, True))
    nernst_manual_layer_averaging(
        "/data/scc/marian.gunsch/AM_tiltedX_Tstep_nernst_T4/spin-configs-99-999/spin-config-99-999-005000.dat",
        title_suffix="Tstep in [110], T=4",
        save_suffix="tilt_xTstep_T4"
    )
    # File not generated yet!
    # nernst_manual_layer_averaging(
    #     "/data/scc/marian.gunsch/08_tilted_yTstep/T4/spin-configs-99-999/spin-config-99-999-005000.dat",
    #     'y',
    #     title_suffix="Tstep in [-110], T=4",
    #     save_suffix="tilt_yTstep_T4"
    # )
    nernst_manual_layer_averaging(
        "/data/scc/marian.gunsch/08_xTstep/T4/spin-configs-99-999/spin-config-99-999-005000.dat",
        is_tilted=False,
        title_suffix="Tstep in [100], T=4",
        save_suffix="xTstep_T4"
    )
    nernst_manual_layer_averaging(
        "/data/scc/marian.gunsch/08_yTstep/T4/spin-configs-99-999/spin-config-99-999-005000.dat",
        'y',
        is_tilted=False,
        title_suffix="Tstep in [010], T=4",
        save_suffix="yTstep_T4"
    )



# %% Main

if __name__ == '__main__':
    # presenting_data_01()

    # presenting_data_02()

    # presenting_data_03()
    # presenting_data_04()

    # presenting_data_05()

    # presenting_data_06()

    # compare_nernst_with_equilibrium((False, True))


    presenting_data_07()

    # presenting_data_08()

    pass


