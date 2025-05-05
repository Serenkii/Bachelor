import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy as sp


def neel_vector(Sz_A, Sz_B):
    """
    Returns the neel vector (value) as an array, where each entry is the value for a specific layer index.
    :param Sz_A: Time averaged z-components of spin vectors, sublattice A.
    :param Sz_B: Time averaged z-components of spin vectors, sublattice B.
    :return: The neel vector (value) of the spin vectors for each layer where the index of the resulting array corresponds to the index of the lattice layer.
    """
    return 0.5 * (Sz_A - Sz_B)


def magnetizazion(Sz_A, Sz_B):
    """
    Returns the magnetization as an array, where each entry is the value for a specific layer index.
    :param Sz_A: Time averaged z-components of spin vectors, sublattice A.
    :param Sz_B: Time averaged z-components of spin vectors, sublattice B.
    :return: The magnetization of the spin vectors for each layer where the index of the resulting array corresponds to the index of the lattice layer.
    """
    return 0.5 * (Sz_A + Sz_B)


# TODO: Test
def spin_currents(spin_x_A, spin_y_A, spin_x_B, spin_y_B):
    """
    Returns different spin currents according to different definitions. The spin currents are returned as an array.
    The array index corresponds to the index of the lattice layer or rather the position between the index and the
    following layer. Therefore, the returned spin current also has one entry less than the passed spins.
    :param spin_x_A:
    :param spin_y_A:
    :param spin_x_B:
    :param spin_y_B:
    :return: j_interSL_+, j_interSL_-, j_intraSL_A, j_intraSL_B, j_ulrike
    """
    ## Following formulas are from paper 'Atomistic spin dynamics simulations of magnonic spin Seebeck
    ## and spin Nernst effects in altermagnets' by Markus Weißenhofer and Alberto Marmadoro

    # inter-sublattice spin current
    j_inter_1 = - (np.average(spin_x_A[:, :-1] * spin_y_A[:, 1:], axis=0)
                   + np.average(spin_y_B[:, :-1] * spin_x_B[:, 1:], axis=0))
    j_inter_2 = - (np.average(spin_x_A[:, :-1] * spin_y_A[:, 1:], axis=0)
                   - np.average(spin_y_B[:, :-1] * spin_x_B[:, 1:], axis=0))

    # intra-sublattice spin current
    j_intra_A = - (np.average(spin_x_A[:, :-1] * spin_y_A[:, 1:], axis=0)
                   - np.average(spin_y_A[:, :-1] * spin_x_A[:, 1:], axis=0))
    j_intra_B = - (np.average(spin_x_B[:, :-1] * spin_y_B[:, 1:], axis=0)
                   - np.average(spin_y_B[:, :-1] * spin_x_B[:, 1:], axis=0))

    ## Following formula is from Ulrike Ritzmann's dissertation, page 86, formula (7.6)
    j_otherpaper = - np.average(spin_x_A[:, :-1] * spin_y_A[:, 1:] - spin_y_B[:, :-1] * spin_x_B[:, 1:], axis=0)

    return j_inter_1, j_inter_2, j_intra_A, j_intra_B, j_otherpaper

