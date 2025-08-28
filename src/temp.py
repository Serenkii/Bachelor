import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy as sp

import mag_util
import bulk_util
import utility
import physics
import plot_util

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

# dataA = np.loadtxt("data/temp/morezlayers/mag-profile-noABC_A.dat")
# dataB = np.loadtxt("data/temp/morezlayers/mag-profile-noABC_B.dat")
# data_string = "Tstep=7meV, more z layers without ABCs, 000"
# save_string_suffix = "morez_noABC"
#
# data_eq_7meV = np.loadtxt("data/temp/altermagnet-equilibrium-7meV.dat")

# %%
#
# spin_x_A, spin_y_A, spin_z_A = mag_util.get_components_as_tuple(dataA, which='xyz', skip_time_steps=1)
# spin_x_B, spin_y_B, spin_z_B = mag_util.get_components_as_tuple(dataB, which='xyz', skip_time_steps=1)
#
# Sz_A = utility.time_avg(spin_z_A)
# Sz_B = utility.time_avg(spin_z_B)
#
# neel = physics.neel_vector(Sz_A, Sz_B)
# magn = physics.magnetizazion(Sz_A, Sz_B)
#
# j_inter_1, j_inter_2, j_intra_A, j_intra_B, j_other = physics.spin_currents(spin_x_A, spin_y_A, spin_x_B, spin_y_B)
#
# plot_util.quick_plot_magn_neel(magn, neel, data_string, delta_x=4)
# plot_util.quick_plot_spin_currents(j_inter_1, j_inter_2, j_intra_A, j_intra_B, j_other, data_string)


# %% equilibriums
# time_steps_for_avg = 100
#
# Sz_A_7eq = np.average(data_eq_7meV[-time_steps_for_avg:, 5], axis=0)
# Sz_B_7eq = np.average(data_eq_7meV[-time_steps_for_avg:, 8], axis=0)
#
# neel_7eq = 0.5 * (Sz_A_7eq - Sz_B_7eq)
# magn_7eq = 0.5 * (Sz_A_7eq + Sz_B_7eq)
#
# neel_0eq = 1
# magn_0eq = 0

# TODO


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


# %% quick diagnostics of spin config file

path = "/data/scc/marian.gunsch/AM_tiltedX_ttmstep_7meV_2_id2/spin-configs-99-999/spin-config-99-999-010000.dat"
data = np.loadtxt(path)

# %%
SL_A = np.where(np.expand_dims(data[:, 3] == 1.0, 1), data[:, :4], np.nan)       # SL A
SL_B = np.where(np.expand_dims(data[:, 3] == 2.0, 1), data[:, :4], np.nan)       # SL A
SL_A_maxs = np.nanmax(SL_A, axis=0)
SL_A_mins = np.nanmin(SL_A, axis=0)
SL_B_maxs = np.nanmax(SL_B, axis=0)
SL_B_mins = np.nanmin(SL_B, axis=0)

# %%

# print("These values seem to show, that there is an effect. Now need to verify, it is not SSE.\n")
#
#     # Comparing with SSE effect
#
#     seebeckT2_path = "/data/scc/marian.gunsch/04_AM_tilted_xTstep_T2-2/spin-configs-99-999/mag-profile-99-999.altermagnetA.dat"
#     mag_util.plot_magnetic_profile_from_paths(
#         [seebeckT2_path,], None, None, None, None, None,
#         [dict(label="T=2meV, seebeck"),],
#         which="z"
#     )
#
#     sse_spin_z_A = mag_util.time_avg(mag_util.get_component(np.loadtxt(seebeckT2_path)))
#     sse_spin_z_B = mag_util.time_avg(mag_util.get_component(np.loadtxt(mag_util.infer_path_B(seebeckT2_path))))
#
#     sse_avg_spin_A = np.mean(sse_spin_z_A[15:-15])
#     sse_avg_spin_B = np.mean(sse_spin_z_B[15:-15])
#
#     print(f"sse_avg_spin_A = {sse_avg_spin_A:.5f} \t sse_avg_spin_B = {sse_avg_spin_B:.5f}")
#     print(f"sne_avg_spin_A = {sne_avg_spin_A:.5f} \t sne_avg_spin_B = {sne_avg_spin_B:.5f}")


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

nx, ny = 512, 512

# Example data: define values on the full checkerboard lattice
x = np.arange(nx)
y = np.arange(ny)
X, Y = np.meshgrid(x, y)

# Data on integer lattice
Z1 = np.sin(X) + np.cos(Y)

# Data on half-integer lattice (offset grid)
xh = np.arange(nx-1) + 0.5
yh = np.arange(ny-1) + 0.5
Xh, Yh = np.meshgrid(xh, yh)
Z2 = np.sin(Xh) + np.cos(Yh)

# Function to build diamonds
def make_diamonds(X, Y, Z):
    polys, colors = [], []
    for (cx, cy, val) in zip(X.ravel(), Y.ravel(), Z.ravel()):
        colors.append(val)
        poly = [(cx, cy+0.5),
                (cx+0.5, cy),
                (cx, cy-0.5),
                (cx-0.5, cy)]
        polys.append(poly)
    return polys, colors

# Build both sets
polys1, colors1 = make_diamonds(X, Y, Z1)
polys2, colors2 = make_diamonds(Xh, Yh, Z2)

# Combine
polys = polys1 + polys2
colors = colors1 + colors2

# Plot
fig, ax = plt.subplots()
collection = PolyCollection(polys, array=np.array(colors),
                            cmap="coolwarm"#, edgecolors="k", linewidth=0.3
                            )
ax.add_collection(collection)
ax.autoscale_view()
ax.set_aspect("equal")

plt.colorbar(collection, ax=ax, label="Value")
plt.show()

