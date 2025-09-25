import sys
import numpy as np
import thesis.mpl_configuration as mpl_configuration

import matplotlib.pyplot as plt

import thesis.subsections.miscellaneous as miscellaneous
import thesis.subsections.equilibrium as equilibrium
import thesis.subsections.sse as sse
import thesis.subsections.sne as sne
import thesis.subsections.dmi as dmi

testing = True

def close_figs():
    if not testing:
        plt.close('all')


def run_spectra(shading):
    print("Running spectra...")
    print(f"Using shading '{shading}'")

    equilibrium.dispersion_comparison_Bfield_table(2, shading)
    close_figs()
    equilibrium.dispersion_comparison_negB(shading)
    close_figs()

    sne.sne_magnon_spectrum()
    close_figs()

    dmi.dispersion_relation_dmi(shading)
    close_figs()


def main():
    pass

    # run_spectra('auto')
    run_spectra('gouraud')

    # miscellaneous.main()

    # equilibrium.main()
    # close_figs()

    # sse.main()
    # close_figs()

    # sne.main()
    # close_figs()

    # dmi.main()
    # close_figs()


if __name__ == '__main__':
    mpl_configuration.configure()

    if len(sys.argv) == 1:
        mpl_configuration.default_configure()
    if len(sys.argv) >= 2:
        backend = sys.argv[1]
        ssh = True if len(sys.argv) == 3 and sys.argv[2] == 'True' else False

        mpl_configuration.configure_backends(backend, ssh)

    main()
