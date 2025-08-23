import numpy as np
import src.spinconf_util as spinconf_util
import src.mag_util as mag_util

import matplotlib.pyplot as plt
import matplotlib.colors as colors


# %% Verifying spin configuration


def check_spin_configuration(tilted):
    path_tilted = "/data/scc/marian.gunsch/08/08_tilted_yTstep/T1/spin-configs-99-999/spin-config-99-999-000000.dat"
    path_nontilted = "/data/scc/marian.gunsch/08/08_yTstep/T1/spin-configs-99-999/spin-config-99-999-000000.dat"

    path = path_tilted if tilted else path_nontilted

    dist = 6
    slice_left = slice(0, dist)
    slice_right = slice(-dist, None)

    data_grid = spinconf_util.read_spin_config_dat_raw(path)

    z = 0
    comp = 2

    xi = np.arange(0, data_grid.shape[0], 1, dtype=int)
    yi = np.arange(0, data_grid.shape[1], 1, dtype=int)
    # X, Y = np.meshgrid(np.concatenate((xi[slice_left], xi[slice_right])),
    #                    np.concatenate((yi[slice_left], yi[slice_right])),
    #                    sparse=True, indexing='xy')

    X, Y = np.meshgrid(np.arange(0, 2 * dist, 1, dtype=int),
                       np.arange(0, 2 * dist, 1, dtype=int),
                       sparse=True, indexing='xy')

    fig, axs = plt.subplots(2)
    title = "tilted" if tilted else "nontilted"
    fig.suptitle(title)

    for SL in [0, 1]:
        axs[SL].set_aspect('equal', 'box')

        data = data_grid[:, :, z, SL, comp]
        data = np.concatenate((data[slice_left], data[slice_right]), axis=0)
        data = np.concatenate((data[:, slice_left], data[:, slice_right]), axis=1)

        im = axs[SL].pcolormesh(X, Y, data.T, norm=colors.CenteredNorm(), cmap='RdBu_r')
        fig.colorbar(im, ax=axs[SL])

        axs[SL].margins(x=0, y=0)
        axs[SL].set_title(f"SL {SL}")

        axs[SL].vlines(dist - 0.5, -0.5, 2 * dist - 0.5, colors="green")
        axs[SL].hlines(dist - 0.5, -0.5, 2 * dist - 0.5, colors="green")

    fig.tight_layout()

    save_suffix = "tilted" if tilted else "nontilted"
    fig.savefig(f"out/miscellaneous/config_{save_suffix}.pdf")

    plt.show()


def confirm_profile_config(tilted, i=0):
    t = 5000
    SL = 0
    component = 2   # z

    conf_path_tilted = f"/data/scc/marian.gunsch/08/08_tilted_yTstep/T1/spin-configs-99-999/spin-config-99-999-00{t}.dat"
    conf_path_nontilt = f"/data/scc/marian.gunsch/08/08_yTstep/T1/spin-configs-99-999/spin-config-99-999-00{t}.dat"

    # average taken over x-axis of sublattice altermagnetA
    prof_path_tilted = "/data/scc/marian.gunsch/08/08_tilted_yTstep/T1/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat"
    prof_path_nontilt = "/data/scc/marian.gunsch/08/08_yTstep/T1/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat"

    conf_path = conf_path_tilted if tilted else conf_path_nontilt
    prof_path = prof_path_tilted if tilted else prof_path_nontilt
    avg_num = 2 if tilted else 1
    i0 = 2 if tilted else 1

    data_A = np.loadtxt(prof_path)

    prof_S0_z = data_A[t, component + 1 + 3 * i]

    data_grid = spinconf_util.read_spin_config_dat_raw(conf_path)
    l = i0 + i * avg_num
    r = l + avg_num
    conf_S0_z = np.nanmean(data_grid[l:r, :, :, SL, component])

    print("")
    print(f"tilted = {tilted}")
    print(f"profile[{i}] = {prof_S0_z}")
    print(f"manual avg w/ conf = {conf_S0_z}")
    print(f"Difference: {prof_S0_z - conf_S0_z}")



# %% Main

def main():
    check_spin_configuration(False)
    check_spin_configuration(True)
    for i in range(4):
        confirm_profile_config(False, i)
        confirm_profile_config(True, i)


if __name__ == '__main__':
    main()
