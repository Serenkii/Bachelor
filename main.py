import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import scipy as sp

import src.utility as util
import src.mag_util as mag_util
import src.bulk_util as bulk_util
import src.utility as util
import src.plot_util as plot_util
import src.physics as physics
import src.spinconf_util as spinconf_util




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
    equi_dmi_path = "/data/scc/marian.gunsch/AM_tiltedX_ttmstairs_DMI-2/AM_Teq-99-999.dat"

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
        magn.append(physics.magnetizazion(spins_A[i], spins_B[i]))
        magn_DMI.append(physics.magnetizazion(spins_DMI_A[i], spins_DMI_B[i]))



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
    magn_dmi = physics.magnetizazion(Sz_dmi_A, Sz_dmi_B)

    neel_nodmi = physics.neel_vector(Sz_nodmi_A, Sz_nodmi_B)
    magn_nodmi = physics.magnetizazion(Sz_nodmi_A, Sz_nodmi_B)

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


# %% Beginning of June

def magnetization_neel_2d_plot():
    print("Showing magnetizazion and Neel vector for a temperature of T=2meV. We are subtracting the equilibrium state."
          " One plot with and one without convolution.\n"
          "Parameters of the simulation: open open open boundaries, 512x512x64")

    path = "/data/scc/marian.gunsch/AM_tiltedX_Tstep_nernst_T2/spin-configs-99-999/spin-config-99-999-005000.dat"
    data = spinconf_util.read_spin_config_dat(path)

    eq_path = "/data/scc/marian.gunsch/AM_tiltedX_ttmstairs_T2meV/spin-configs-99-999/spin-config-99-999-005000.dat"
    eq_data = spinconf_util.read_spin_config_dat(eq_path)

    print(f"Finished reading data...\n"
          f"Data path: {path}\n"
          f"Equilibrium path: {eq_path}\n")

    magn, neel = spinconf_util.calculate_magnetization_neel(data, eq_data, rel_Tstep_pos=0.49)

    magn_zavg = spinconf_util.average_z_layers(magn["z"])
    neel_zavg = spinconf_util.average_z_layers(neel["z"])

    spinconf_util.plot_colormap(magn_zavg, title="magnetization, equi subtracted (T=2meV)")
    spinconf_util.plot_colormap(neel_zavg, title="Neel vector (z), equi subtracted (T=2meV)")

    spinconf_util.plot_colormap(spinconf_util.convolute(magn_zavg),
                                title="magnetization, equi subtracted (T=2meV) - convoluted")
    spinconf_util.plot_colormap(spinconf_util.convolute(neel_zavg),
                                title="Neel vector (z), equi subtracted (T=2meV) - convoluted")


def spin_currents_2d_plot():
    print("I attempted to gain useful data by calculating the spin currents. Luckily, you can clearly see the Spin "
          "Seebeck effect when looking at the longitudinal direction of the spin currents. Sadly, for Spin Nernst, "
          "not much is visible. The following will be plotted: 1. spin currents for SSE, 2. spin currents for SNE, "
          "3. spin currents for SNE but convoluted.")

    data_path = "/data/scc/marian.gunsch/AM_tiltedX_ttmstep_morezlayers_noABC/spin-configs-99-999/spin-config-99-999-005000.dat"
    # data_path = "/data/scc/marian.gunsch/01_AM_tilted_Tstep/spin-configs-99-999/spin-config-99-999-005000.dat"

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

        plt.show()

        # TODO: print statement on what was plotted and saving figure


def fourier_thingy_TODO_CHANGENAME():
    print("Lorem ipsum TODO")


def presenting_data_02():
    seperator = "-------------------------------------------------------------\n"
    print("[05.06.25] Presenting data for next meeting with Uli. We are talking about/showing the following:"
          "We want to show the 2D-plots for the whole spin configuration. We want to show SNE, SSE and the Fourier "
          "analysis.\n"
          + seperator)

    # magnetization_neel_2d_plot()
    # print(seperator)

    spin_currents_2d_plot()
    print(seperator)

    # fourier_thingy_TODO_CHANGENAME()
    # print(seperator)



# %% Main

if __name__ == '__main__':
    # temperature_dependent_nernst(save=True, save_path='out/T-dependent-nernst.png', delta_x=0)
    # dmi_ground_state_comparison(save=True, save_path='out/ground_state_comparison_DMI.pdf')

    # quick_seebeck_dmi_comparison()
    # quick_nernst_dmi_comparison()

    presenting_data_02()


