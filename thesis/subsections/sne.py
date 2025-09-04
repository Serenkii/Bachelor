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


# %% Spin accumulation / net magnetization

# See here e.g. for T-dependence: /data/scc/marian.gunsch/00/AM_tiltedX_Tstep_nernst/

def sne_spin_accumulation():
    temperature = 2     # available: 1..10
    T = temperature

    paths = {
        "100": f"/data/scc/marian.gunsch/08/08_xTstep/T{T}/",
        "010": f"/data/scc/marian.gunsch/08/08_yTstep/T{T}/",
        "110": f"/data/scc/marian.gunsch/00/AM_tiltedX_Tstep_nernst/AM_tiltedX_Tstep_nernst_T{T}/",
        "-110": f"/data/scc/marian.gunsch/08/08_tilted_yTstep/T{T}/",
    }

    # for cold region
    equi_T0_paths = {
        "100": "/data/scc/marian.gunsch/12/AM_Tstairs_x_T0_openbou/",
        "010": "/data/scc/marian.gunsch/12/AM_Tstairs_y_T0_openbou/",
        "110": "/data/scc/marian.gunsch/12/AM_tilt_Tstairs_x_T0_openbou/",
        "-110": "/data/scc/marian.gunsch/12/AM_tilt_Tstairs_y_T0_openbou/"
    }

    mag_util.npy_files_from_dict(paths)
    mag_util.npy_files_from_dict(equi_T0_paths)


# %% Spin currents (transversal)

def spin_currents_open():
    # measurement direction (profile axis)
    paths = {
        "100": "/data/scc/marian.gunsch/11/AM_xTstep_y/",
        "010": "/data/scc/marian.gunsch/11/AM_yTstep_x/",
        "-110": "/data/scc/marian.gunsch/11/AM_tilt_xTstep_y/",
        "110": "/data/scc/marian.gunsch/11/AM_tilt_yTstep_x/"
    }
    # Direction of the temperature step
    step_dir = {
        "010": "100",
        "100": "010",
        "-110": "110",
        "110": "-110"
    }

    mag_util.npy_files_from_dict(paths)


def spin_currents_upperABC():
    pass
    print("Massive problems running simulations.")
    print("Jobscripts labelled with '11' ")


def spin_currents_uploABC():
    pass
    print("Massive problems running simulations.")
    print("Jobscripts labelled with '11' ")


# %% Magnon spectrum

def sne_magnon_spectrum():
    # direction of measurement (profile axis)
    paths = {
        "-110": "/data/scc/marian.gunsch/02/02_AM_tilted_Tstep_hightres/",
        "110": "/data/scc/marian.gunsch/14/AM_tilt_yTstep_x_hightres/"
    }
    # Direction of the temperature step
    step_dir = {
        "-110": "110",
        "110": "-110"
    }

    mag_util.npy_files_from_dict(paths)


# %% Main

def main():
    sne_spin_accumulation()
    spin_currents_open()
    spin_currents_upperABC()
    spin_currents_uploABC()
    sne_magnon_spectrum()
