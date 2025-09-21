import sys
import numpy as np
import thesis.mpl_configuration as mpl_configuration

import matplotlib.pyplot as plt

import thesis.subsections.miscellaneous as miscellaneous
import thesis.subsections.equilibrium as equilibrium
import thesis.subsections.sse as sse
import thesis.subsections.sne as sne
import thesis.subsections.dmi as dmi


def run_spectra():
    print("Running spectra...")

    equilibrium.dispersion_comparison_Bfield_table(2)
    equilibrium.dispersion_comparison_negB()

    sne.sne_magnon_spectrum()



def main():
    pass

    # run_spectra()

    # miscellaneous.main()

    equilibrium.main()
    sse.main()
    sne.main()
    dmi.main()


if __name__ == '__main__':
    mpl_configuration.configure()

    if len(sys.argv) == 1:
        mpl_configuration.default_configure()
    if len(sys.argv) >= 2:
        backend = sys.argv[1]
        ssh = True if len(sys.argv) == 3 and sys.argv[2] == 'True' else False

    main()
