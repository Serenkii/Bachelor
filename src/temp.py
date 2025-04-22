import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy as sp


# %%
N = 512


# %%

dataA = np.loadtxt("data/temp/mag-profile-99-999.altermagnetA.dat")
# dataA = np.loadtxt("data/temp/mag-profile-99-999.altermagnetA-3.dat")
dataB = np.loadtxt("data/temp/mag-profile-99-999.altermagnetB.dat")
# dataB = np.loadtxt("data/temp/mag-profile-99-999.altermagnetB-3.dat")

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

fig, ax = plt.subplots()
ax.set_title("Neel")
ax.plot(neel)

plt.show()

fig, ax = plt.subplots()
ax.set_title("Magn")
ax.plot(magn)

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

# %%

fig, ax = plt.subplots()
ax.set_title("j_inter_+")
ax.plot(j_inter_1)

plt.show()

fig, ax = plt.subplots()
ax.set_title("j_inter_-")
ax.plot(j_inter_2)

plt.show()

fig, ax = plt.subplots()
ax.set_title("j_intra_A")
ax.plot(j_intra_A)

plt.show()

fig, ax = plt.subplots()
ax.set_title("j_intra_B")
ax.plot(j_intra_B)

plt.show()

# %% quick and ugly plotting of time dependent for equilibrium

data_quick_DMI = np.loadtxt("data/temp/AM_DMI_quick_avg.dat")

time = data_quick_DMI[:, 0]
sx = data_quick_DMI[:, 3]
sy = data_quick_DMI[:, 4]
sz = data_quick_DMI[:, 5]

fig, ax = plt.subplots()

ax.set_title("DMI")
ax.plot(time, sx, label="sx")
ax.plot(time, sy, label="sy")

ax.legend()

plt.show()

fig, ax = plt.subplots()
ax.set_title("DMI, sz")
ax.plot(time, sz, label="sz")

plt.show()



# %%
data_DMI_equi_ferri = np.loadtxt("/data/scc/marian.gunsch/AM_tiltedX_ttmstairs_DMI_ferri-2/AM_Teq-99-999.dat")
data_DMI_equi = np.loadtxt("/data/scc/marian.gunsch/AM_tiltedX_ttmstairs_DMI-2/AM_Teq-99-999.dat")
data_DMI_equi_ferri_large = np.loadtxt("/data/scc/marian.gunsch/AM_tiltedX_ttmstairs_DMI_large_ferri/AM_Teq-99-999.dat")
data_DMI_equi_large = np.loadtxt("/data/scc/marian.gunsch/AM_tiltedX_ttmstairs_DMI_large/AM_Teq-99-999.dat")

fig, ax = plt.subplots()
temp = data_DMI_equi_ferri
ax.set_title("DMI equi ferri")
ax.plot(temp[:, 0], temp[:,3], label="sx")
ax.plot(temp[:, 0], temp[:,4], label="sy")
ax.plot(temp[:, 0], temp[:,5], label="sz")
ax.legend()
plt.show()

fig, ax = plt.subplots()
temp = data_DMI_equi
ax.set_title("DMI equi")
ax.plot(temp[:, 0], temp[:,3], label="sx")
ax.plot(temp[:, 0], temp[:,4], label="sy")
ax.plot(temp[:, 0], temp[:,5], label="sz")
ax.legend()
plt.show()

fig, ax = plt.subplots()
temp = data_DMI_equi_ferri_large
ax.set_title("DMI equi ferri large")
ax.plot(temp[:, 0], temp[:,3], label="sx")
ax.plot(temp[:, 0], temp[:,4], label="sy")
ax.plot(temp[:, 0], temp[:,5], label="sz")
ax.legend()
plt.show()

fig, ax = plt.subplots()
temp = data_DMI_equi_large
ax.set_title("DMI equi large")
ax.plot(temp[:, 0], temp[:,3], label="sx")
ax.plot(temp[:, 0], temp[:,4], label="sy")
ax.plot(temp[:, 0], temp[:,5], label="sz")
ax.legend()
plt.show()

# %%
import utility as util
util.plot_spin_xyz_over_t("/data/scc/marian.gunsch/AM_tiltedX_ttmstairs_DMI_ferri-2/AM_Teq-99-999.dat", "DMI ferri")
