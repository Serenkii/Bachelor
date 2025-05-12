import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy as sp

import src.utility as util
import src.mag_util as mag_util
import src.bulk_util as bulk_util
import src.utility as util
import src.plot_util as plot_util
import src.physics as physics

Nx = 512
Ny = Nx
Nz = 2
temperature_step_rel_pos = 0.49


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



def dmi_ground_state_comparison():
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
    markers = ['o', 's', '^', 'D']  # Circle, square, triangle, diamond
    colors = ['red', 'green', 'blue', 'purple']

    # Plot each type across all groups (x, y, z)
    for i, t in enumerate(types):
        group_labels = list(data.keys())  # ['x', 'y', 'z']
        values = [data[label][i] for label in group_labels]  # Get i-th value for each group
        ax.scatter(group_labels, values, label=f'Type {t}', marker=markers[i], color=colors[i])

    # Labeling
    ax.set_ylabel('Average value')
    ax.set_title('Equilibrium 2meV: Spin components for different sublattices with and without DMI')
    ax.legend()

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
        fig.savefig(f"{save_path}")

    plt.show()


def quick_seebeck_dmi_comparison():
    print("Comparing the spin Seebeck effect with and without antisymmetric exchange at a temperature of 2 meV.")
    dmi_comparison(
        "/data/scc/marian.gunsch/AM-DMI_tilted_Tstep_seebeck/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat",
        "/data/scc/marian.gunsch/AM-DMI_tilted_Tstep_seebeck/spin-configs-99-999/mag-profile-99-999.altermagnetB.dat",
        "/data/scc/marian.gunsch/AM-tilted_Tstep_seebeck/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat",
        "/data/scc/marian.gunsch/AM-tilted_Tstep_seebeck/spin-configs-99-999/mag-profile-99-999.altermagnetB.dat",
        "Comparison Seebeck (no equilibriums were subtracted!)",
        10,
        "out/comparison_DMI_seebeck_.png"
    )


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
        "out/comparison_DMI_nernst_.png"
    )



if __name__ == '__main__':
    #    temperature_dependent_nernst(save=True, save_path='out/T-dependent-nernst.png', delta_x=0)
    dmi_ground_state_comparison()

    quick_seebeck_dmi_comparison()
    quick_nernst_dmi_comparison()
