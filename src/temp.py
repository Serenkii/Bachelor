import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy as sp


# %%
N = 512


# %%
data_string = ""

# dataA = np.loadtxt("data/temp/mag-profile-99-999.altermagnetA.dat")
# dataA = np.loadtxt("data/temp/mag-profile-99-999.altermagnetA-3.dat")
# dataB = np.loadtxt("data/temp/mag-profile-99-999.altermagnetB.dat")
# dataB = np.loadtxt("data/temp/mag-profile-99-999.altermagnetB-3.dat")

# more z layers quick and dirty...
# dataA = np.loadtxt("data/temp/morezlayers/mag-profile-ABC_A.dat")
# dataB = np.loadtxt("data/temp/morezlayers/mag-profile-ABC_B.dat")
# data_string = "Tstep=7meV, more z layers with ABC in upper y, 000"
# save_string_suffix = "morez_ABC"

dataA = np.loadtxt("data/temp/morezlayers/mag-profile-noABC_A.dat")
dataB = np.loadtxt("data/temp/morezlayers/mag-profile-noABC_B.dat")
data_string = "Tstep=7meV, more z layers without ABCs, 000"
save_string_suffix = "morez_noABC"

data_eq_7meV = np.loadtxt("data/temp/altermagnet-equilibrium-7meV.dat")

# %% select z components

# Select all z component of spins, select all time steps except the very first
# because its values are weirdly a little bit higher
spin_z_A = dataA[1:, 3::3]
spin_z_B = dataB[1:, 3::3]

# the following does not make any sense because we are measuring along y-axis and not x

# %% select other components
spin_x_A = dataA[1:, 1::3]
spin_y_A = dataA[1:, 2::3]
spin_x_B = dataB[1:, 1::3]
spin_y_B = dataB[1:, 2::3]

# %% time average

Sz_A = np.average(spin_z_A, axis=0)
Sz_B = np.average(spin_z_B, axis=0)

# %% neel vector and total magnetisation (z component)
neel = 0.5 * (Sz_A - Sz_B)
magn = 0.5 * (Sz_A + Sz_B)

# %% equilibriums
time_steps_for_avg = 100

Sz_A_7eq = np.average(data_eq_7meV[-time_steps_for_avg:, 5], axis=0)
Sz_B_7eq = np.average(data_eq_7meV[-time_steps_for_avg:, 8], axis=0)

neel_7eq = 0.5 * (Sz_A_7eq - Sz_B_7eq)
magn_7eq = 0.5 * (Sz_A_7eq + Sz_B_7eq)

neel_0eq = 1
magn_0eq = 0

# TODO

# %% subtracting equlibrium

# TODO

# %% Plot neel and magn
delta = 4
x = np.arange(delta, neel.size - delta, 1.0)

fig, ax = plt.subplots()
ax.set_xlabel("position (index)")
ax.set_ylabel("magnitude (au)")
ax.set_title(f"Neel (~SzA-SzB) ({data_string})")
ax.plot(x, neel[delta:-delta])
plt.savefig(f"out/Neel_{save_string_suffix}.png")
plt.show()

fig, ax = plt.subplots()
ax.set_xlabel("position (index)")
ax.set_ylabel("magnitude (au)")
ax.set_title(f"Magn (~SzA+SzB) ({data_string})")
ax.plot(x, magn[delta:-delta])
plt.savefig(f"out/Magn_{save_string_suffix}.png")
plt.show()


# %% spin current

# intersublattice spin current
j_inter_1 = - (np.average(spin_x_A[:, :-1] * spin_y_A[:, 1:], axis=0)
             + np.average(spin_y_B[:, :-1] * spin_x_B[:, 1:], axis=0))
j_inter_2 = - (np.average(spin_x_A[:, :-1] * spin_y_A[:, 1:], axis=0)
             - np.average(spin_y_B[:, :-1] * spin_x_B[:, 1:], axis=0))

# inrtrasublattice spin current
j_intra_A = - (np.average(spin_x_A[:, :-1] * spin_y_A[:, 1:], axis=0)
             - np.average(spin_y_A[:, :-1] * spin_x_A[:, 1:], axis=0))
j_intra_B = - (np.average(spin_x_B[:, :-1] * spin_y_B[:, 1:], axis=0)
             - np.average(spin_y_B[:, :-1] * spin_x_B[:, 1:], axis=0))

# other paper, other formula (Ulrike, p.86)
j_otherpaper = - np.average(spin_x_A[:, :-1] * spin_y_A[:, 1:] - spin_y_B[:, :-1] * spin_x_B[:, 1:], axis=0)


# %%

fig, ax = plt.subplots()
ax.set_title(f"Spin currents ({data_string})")
ax.set_xlabel("spin index = position")
ax.set_ylabel("magnitude [au]")
ax.plot(j_inter_1, label="j_inter_+", linewidth=0.6)
ax.plot(j_inter_2, label="j_inter_-", linewidth=0.6)
ax.plot(j_intra_A, label="j_intra_A", linewidth=0.6)
ax.plot(j_intra_B, label="j_intra_B", linewidth=0.6)
ax.plot(j_otherpaper, label="j_otherpaper", linewidth=0.8)
ax.legend()
plt.savefig(f"out/spin_current_{save_string_suffix}.png")
plt.show()

# %% quick and ugly plotting of time dependent for equilibrium

# data_quick_DMI = np.loadtxt("data/temp/AM_DMI_quick_avg.dat")
#
# time = data_quick_DMI[:, 0]
# sx = data_quick_DMI[:, 3]
# sy = data_quick_DMI[:, 4]
# sz = data_quick_DMI[:, 5]
#
# fig, ax = plt.subplots()
#
# ax.set_title("DMI")
# ax.plot(time, sx, label="sx")
# ax.plot(time, sy, label="sy")
#
# ax.legend()
#
# plt.show()
#
# fig, ax = plt.subplots()
# ax.set_title("DMI, sz")
# ax.plot(time, sz, label="sz")
#
# plt.show()



# %%
import utility as util
# util.plot_spin_xyz_over_t("/data/scc/marian.gunsch/AM_tiltedX_ttmstairs_DMI_ferri-2/AM_Teq-99-999.dat",
#                           "DMI equi ferri")
#
# util.plot_spin_xyz_over_t("/data/scc/marian.gunsch/AM_tiltedX_ttmstairs_DMI-2/AM_Teq-99-999.dat",
#                           "DMI equi")
#
# util.plot_spin_xyz_over_t("/data/scc/marian.gunsch/AM_tiltedX_ttmstairs_DMI_large_ferri/AM_Teq-99-999.dat",
#                           "DMI equi ferri large")
#
# util.plot_spin_xyz_over_t("/data/scc/marian.gunsch/AM_tiltedX_ttmstairs_DMI_large/AM_Teq-99-999.dat",
#                           "DMI equi large")


