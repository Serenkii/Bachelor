import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.colors as colors
from matplotlib.collections import PolyCollection

import src.utility as util
import src.mag_util as mag_util
import src.bulk_util as bulk_util
import src.utility as util
import src.plot_util as plot_util
import src.physics as physics
import src.spinconf_util as spinconf_util
import src.helper as helper


# %%
save_base_path = "out/thesis/sse/"


shared_kwargs = dict(markersize=0.5, linewidth=1.0)
plot_kwargs_dict = {
    "110": dict(**shared_kwargs, marker="", linestyle="-"),     # marker="s"
    "-110": dict(**shared_kwargs, marker="", linestyle="--")     # marker="D"
}

paths = {
    "110": dict(),
    "-110": dict()
}

for B in range(50, 100 + 1, 10):
    paths["110"][B] = f"/data/scc/marian.gunsch/05/05_AM_tilted_xTstep_T2_B{B}/"
    paths["-110"][B] = f"/data/scc/marian.gunsch/05/05_AM_tilted_yTstep_T2_B{B}/"

paths["110"][0] = "/data/scc/marian.gunsch/04/04_AM_tilted_xTstep_T2-2/"
paths["-110"][0] = "/data/scc/marian.gunsch/04/04_AM_tilted_yTstep_T2-2/"
paths["110"][-100] = "/data/scc/marian.gunsch/06/06_AM_tilted_xTstep_T2_Bn100/"
paths["-110"][-100] = "/data/scc/marian.gunsch/06/06_AM_tilted_yTstep_T2_Bn100/"

B_fields = paths["110"].keys()
if B_fields != paths["-110"].keys():
    raise AttributeError("Conflicting keys!")

directions = paths.keys()
field_strengths = sorted(paths["110"].keys(), reverse=True)
assert field_strengths == sorted(paths["-110"].keys(), reverse=True)

data_A = {
    "110": dict(),
    "-110": dict()
}
data_B = {
    "110": dict(),
    "-110": dict()
}

magnetization = {
    "110": dict(),
    "-110": dict()
}

def initialize_data():
    data_A["110"], data_B["110"] = mag_util.npy_files_from_dict(paths["110"])
    data_A["-110"], data_B["-110"] = mag_util.npy_files_from_dict(paths["-110"])
    for direction in directions:
        for B in field_strengths:
            magnetization[direction][B] = physics.magnetization(
                mag_util.get_component(data_A[direction][B], "z", 15),
                mag_util.get_component(data_B[direction][B], "z", 15), True)

# %% INTRODUCTION OF A STATIC MAGNETIC FIELD (MAGNETIZATION)

def plot_magn_Bfield(magn_dict, ylabel=r"magnetization $\langle m \rangle$", xlim=(0, 255)):

    fig, ax = plt.subplots()

    max_val = - np.inf
    min_val = np.inf

    colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']

    lines = dict((direction, []) for direction in directions)

    for direction in directions:
        for B, color in zip(field_strengths, colors):
            # label = r"$\num{" + str(B) + r"}{}$" if direction == "110" else None
            label = r"$\num{" + str(B) + r"}{}$"
            _line, = ax.plot(magn_dict[direction][B], **plot_kwargs_dict[direction],
                            label=label, color=color)
            lines[direction].append(_line)

            max_val = max(max_val, np.max(magn_dict[direction][B][15:-15]))
            min_val = min(min_val, np.min(magn_dict[direction][B][15:-15]))

    val_range = max_val - min_val
    pad = 0.05

    ax.set_xlabel(r"position $x/\tilde{a}$")
    ax.set_ylabel(ylabel)

    ax.set_xlim(*xlim)
    ax.set_ylim(min_val - pad * val_range, max_val + pad * val_range)

    legend1 = plt.legend(handles=lines["110"], ncols=2, title=f"\\hkl[110]: " + r"$B$ (\si{\tesla})",
                         fontsize="small", loc="upper right")
    legend2 = plt.legend(handles=lines["-110"], ncols=2, title=f"\\hkl[-110]: " + r"$B$ (\si{\tesla})",
                         fontsize="small", loc="lower right")
    ax.add_artist(legend1)
    ax.add_artist(legend2)

    dT_pt, = ax.plot(helper.get_Tstep_pos(0.49, 256), min_val - pad * val_range, marker=10,
                     color="r", linestyle="", label=r"$\Delta T$")
    ax.plot(helper.get_Tstep_pos(0.49, 256), max_val + pad * val_range, marker=11, color="r",
            linestyle="",)

    legend_dT = plt.legend(handles=[dT_pt,], loc="lower left")
    ax.add_artist(legend_dT)

    fig.tight_layout()

    # ax.set_xticks(list(ax.get_xticks()) + [helper.get_Tstep_pos(0.49, 256),],
    #               list(ax.get_xticklabels()) + ["$\Delta T$",])


    return fig, ax



# %% magnetization for different magnetic fields
def sse_magnetization_Bfield():
    fig, ax = plot_magn_Bfield(magnetization)

    fig.savefig(f"{save_base_path}magnetization_Bfield.pdf")
    plt.show()



# %% magnon accumulation for different magnetic fields
def sse_magnaccum_Bfield():

    equi_slice = slice(9, 40)

    equilibrium_magn = dict()

    for B in field_strengths:
        l = []
        for direction in directions:
            l.append(np.mean(magnetization[direction][B][equi_slice]))
        equilibrium_magn[B] = np.mean(np.array(l))

    step_pos = helper.get_absolute_T_step_index(0.49, 256)

    magnon_accumulation = {
        "110": dict(),
        "-110": dict()
    }

    for direction in directions:
        for B in field_strengths:
            magnon_accumulation[direction][B] = np.copy(magnetization[direction][B])
            magnon_accumulation[direction][B][:step_pos] -= equilibrium_magn[B]

    fig, ax = plot_magn_Bfield(magnon_accumulation, r"magnon accum. $\langle \Delta m \rangle$", (60, 215))
    fig.savefig(f"{save_base_path}magnonaccum_Bfield.pdf")
    plt.show()

    fig, ax = plot_magn_Bfield(magnon_accumulation, r"magnon accum. $\langle \Delta m \rangle$", (115, 135))
    fig.savefig(f"{save_base_path}magnonaccum_Bfield_zoom__.pdf")
    plt.show()


# %% Plot peaks vs magnetic field
def peak_dependence():
    area_slice = slice(helper.get_index_first_cold(0.49, 256), None)

    abs_max_func = {
        "110": np.max,
        "-110": np.min
    }
    peaks = {
        "110": dict(),
        "-110": dict()
    }

    for direction in directions:
        for B in field_strengths:
            peaks[direction][B] = abs_max_func[direction](magnetization[direction][B][area_slice])

    # Plot

    fig, ax = plt.subplots()

    ax.set_xlabel(r"magnetic field strength $B$ (\si{\tesla})")
    ax.set_ylabel(r"abs. peak magnon accum. $\left\lvert \langle \Delta m \rangle \right\rvert_{\max}$")

    ax.plot(field_strengths, [peaks["110"][B] for B in field_strengths], label=r"\hkl[110]", linestyle="-", marker="o")
    ax.plot(field_strengths, [-peaks["-110"][B] for B in field_strengths], label=r"\hkl[-110]", linestyle="-", marker="o")

    ax.legend(loc="upper center")

    fig.tight_layout()

    fig.savefig(f"{save_base_path}sse_peak_dependence.pdf")

    plt.show()



# %% Compare for different directions for one magnetic field strength: magnetization for [100], [110], [-110]
def direction_comparison():
    path_100 = "/data/scc/marian.gunsch/13/13_AM_xTstep_T2_B100/"

    mag_util.npy_files(path_100, return_data=False)


# %% Propagation lengths for [110] and [-110] for different B-fields
def propagation_lengths():
    start = helper.get_index_first_cold(0.49, 256) + 2
    fit_area = slice(start, start + 60)




# %% SPIN CURRENTS
def sse_spin_currents():

    paths = {
        "100": "/data/scc/marian.gunsch/16/AM_xTstep_T2/",
        "010": "/data/scc/marian.gunsch/16/AM_yTstep_T2/",          # Not sure whether to use
        "110": "/data/scc/marian.gunsch/04/04_AM_tilted_xTstep_DMI_T2-2/",
        "-110": "/data/scc/marian.gunsch/04/04_AM_tilted_yTstep_DMI_T2-2/"
    }

    mag_util.npy_files_from_dict(paths)


# %% Main

def main():
    initialize_data()

    sse_magnetization_Bfield()
    sse_magnaccum_Bfield()

    peak_dependence()
