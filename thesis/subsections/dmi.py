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

# %% EQUILIBRIUM PROPERTIES

def average_spin_components():
    # TODO!
    paths = {
        None: None
    }
    # Think about what exactly to compare? Dmi/no DMI for different temperatures? Or just one temperature?


def dispersion_relation_dmi():
    # TODO!
    paths = {
        "100": "/data/scc/marian.gunsch/15/AM_DMI_Tstairs_T2_x/",     # not sure whether to use
        "010": "/data/scc/marian.gunsch/15/AM_DMI_Tstairs_T2_y/",     # not sure whether to use
        "110": "/data/scc/marian.gunsch/15/AM_tilt_DMI_Tstairs_T2_x/",
        "-110": "/data/scc/marian.gunsch/15/AM_tilt_DMI_Tstairs_T2_y/"
    }

    paths_nodmi = {
        "100": f"/data/scc/marian.gunsch/10/AM_Tstairs_T2_x-2/",      # not sure whether to use
        "010": f"/data/scc/marian.gunsch/10/AM_Tstairs_T2_y-2/",      # not sure whether to use
        "110": f"/data/scc/marian.gunsch/10/AM_tilt_Tstairs_T2_x-2/",
        "-110": f"/data/scc/marian.gunsch/10/AM__tilt_Tstairs_T2_y-2/"  # oups: '__'
    }

    directions = paths.keys()
    if directions != paths.keys():
        raise AttributeError("Conflicting keys!")

    mag_util.npy_files_from_dict(paths)
    mag_util.npy_files_from_dict(paths_nodmi)



# %% SPIN SEEBECK EFFECT

def sse_magnon_accumulation_dmi():
    # TODO!
    paths = {
        "100": "/data/scc/marian.gunsch/16/AM_xTstep_DMI_T2/",
        "010": "/data/scc/marian.gunsch/16/AM_yTstep_DMI_T2/",
        "110": "/data/scc/marian.gunsch/04/04_AM_tilted_xTstep_DMI_T2-2/",
        "-110": "/data/scc/marian.gunsch/04/04_AM_tilted_yTstep_DMI_T2-2/"
    }

    paths_nodmi = {
        "100": "/data/scc/marian.gunsch/16/AM_xTstep_T2/",
        "010": "/data/scc/marian.gunsch/16/AM_yTstep_T2/",
        "110": "/data/scc/marian.gunsch/04/04_AM_tilted_xTstep_T2-2/",
        "-110": "/data/scc/marian.gunsch/04/04_AM_tilted_yTstep_T2-2/"
    }

    mag_util.npy_files_from_dict(paths)
    mag_util.npy_files_from_dict(paths_nodmi)



# %% SPIN SEEBECK EFFECT DMI + MAGNETIC FIELD

def sse_magnon_accumulation_dmi_B():
    # TODO!
    paths = {
        "100": "/data/scc/marian.gunsch/16/AM_xTstep_DMI_B_T2/",
        "010": "/data/scc/marian.gunsch/16/AM_yTstep_DMI_B_T2/",
        "110": "/data/scc/marian.gunsch/05/05_AM_tilted_xTstep_DMI_T2_staticB/",
        "-110": "/data/scc/marian.gunsch/05/05_AM_tilted_yTstep_DMI_T2_staticB/"
    }

    mag_util.npy_files_from_dict(paths)


# %% Main

def main():
    average_spin_components()
    dispersion_relation_dmi()
    sse_magnon_accumulation_dmi()
    sse_magnon_accumulation_dmi_B()