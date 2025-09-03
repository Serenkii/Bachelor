import numpy as np


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
        # "100": "/data/scc/marian.gunsch/15/AM_DMI_Tstairs_T2_x/",
        # "010": "/data/scc/marian.gunsch/15/AM_DMI_Tstairs_T2_y/",
        "110": "/data/scc/marian.gunsch/15/AM_tilt_DMI_Tstairs_T2_x/",
        "-110": "/data/scc/marian.gunsch/15/AM_tilt_DMI_Tstairs_T2_y/"
    }

    paths_nodmi = {
        # "100": f"/data/scc/marian.gunsch/10/AM_Tstairs_T2_x-2/",
        # "010": f"/data/scc/marian.gunsch/10/AM_Tstairs_T2_y-2/",
        "110": f"/data/scc/marian.gunsch/10/AM_tilt_Tstairs_T2_x-2/",
        "-110": f"/data/scc/marian.gunsch/10/AM__tilt_Tstairs_T2_y-2/"  # oups: '__'
    }

    directions = paths.keys()
    if directions != paths.keys():
        raise AttributeError("Conflicting keys!")



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




# %% SPIN SEEBECK EFFECT DMI + MAGNETIC FIELD

def sse_magnon_accumulation_dmi_B():
    # TODO!
    paths = {
        "100": "/data/scc/marian.gunsch/16/AM_xTstep_DMI_B_T2/",
        "010": "/data/scc/marian.gunsch/16/AM_yTstep_DMI_B_T2/",
        "110": "/data/scc/marian.gunsch/05/05_AM_tilted_xTstep_DMI_T2_staticB/",
        "-110": "/data/scc/marian.gunsch/05/05_AM_tilted_yTstep_DMI_T2_staticB/"
    }


# %% Main

def main():
    pass

