import numpy as np


# %% INTRODUCTION OF A STATIC MAGNETIC FIELD (MAGNETIZATION)

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


# %% magnetization for different magnetic fields
def sse_magnetization_Bfield():
    pass

# %% magnon accumulation for different magnetic fields
def sse_magnaccum_Bfield():
    pass


# %% Plot peaks vs magnetic field
def peak_dependence():
    pass


# %% Compare for different directions for one magnetic field strength: magnetization for [100], [110], [-110]
def direction_comparison():
    path_100 = "/data/scc/marian.gunsch/13/13_AM_xTstep_T2_B100/"
    path_100 = "/data/scc/marian.gunsch/13_AM_xTstep_T2_B100/"
    print("Move folder!")


# %% Propagation lengths for [110] and [-110] for different B-fields
def propagation_lengths():
    pass


# %% SPIN CURRENTS
def sse_spin_currents():

    paths = {
        "100": "/data/scc/marian.gunsch/16/AM_xTstep_T2/",
        # "010": "/data/scc/marian.gunsch/16/AM_yTstep_T2/",
        "110": "/data/scc/marian.gunsch/04/04_AM_tilted_xTstep_DMI_T2-2/",
        "-110": "/data/scc/marian.gunsch/04/04_AM_tilted_yTstep_DMI_T2-2/"
    }



# %% Main

def main():
    pass

