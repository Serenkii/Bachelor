import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy as sp

import src.utility as util
import src.mag_util as mag_util
import src.bulk_util as bulk_util

import src.helper as helper

lattice_constant = 1
grid_constant = lattice_constant
grid_constant_tilted = grid_constant / np.sqrt(2)
lattice_constant_tilted = 2 * grid_constant_tilted


def neel_vector(Sz_A, Sz_B, do_time_avg=False):
    """
    Returns the neel vector (value) as an array, where each entry is the value for a specific layer index.
    :param Sz_A: Time averaged z-components of spin vectors, sublattice A.
    :param Sz_B: Time averaged z-components of spin vectors, sublattice B.
    :return: The neel vector (value) of the spin vectors for each layer where the index of the resulting array corresponds to the index of the lattice layer.
    """
    if do_time_avg:
        return 0.5 * (util.time_avg(Sz_A) - util.time_avg(Sz_B))
    return 0.5 * (Sz_A - Sz_B)


def magnetization(Sz_A, Sz_B, do_time_avg=False):
    """
    Returns the magnetization as an array, where each entry is the value for a specific layer index.
    :param Sz_A: Time averaged z-components of spin vectors, sublattice A.
    :param Sz_B: Time averaged z-components of spin vectors, sublattice B.
    :return: The magnetization of the spin vectors for each layer where the index of the resulting array corresponds to the index of the lattice layer.
    """
    if do_time_avg:
        return 0.5 * (util.time_avg(Sz_A) + util.time_avg(Sz_B))
    return 0.5 * (Sz_A + Sz_B)


def index_to_position(array, tilted):
    if tilted:
        return array * lattice_constant_tilted
    return array * lattice_constant


def dispersion(Sx, Sy, dx=lattice_constant, dt=50e-16, factor=2*np.pi):
    Sp = Sx + 1j * Sy

    k_vectors_ = np.fft.fftfreq(Sp.shape[1], d=dx) * factor
    freqs_ = np.fft.fftfreq(Sp.shape[0], d=dt) * factor

    Sp_F_ = np.fft.fft2(Sp)

    Sp_F = np.fft.fftshift(Sp_F_)
    k_vectors = np.fft.fftshift(k_vectors_)
    freqs = np.fft.fftshift(freqs_)

    magnon_density = np.abs(Sp_F) ** 2

    return k_vectors, freqs, magnon_density


def dispersion_from_path(pathA, pathargv, tilted: bool, time_steps=100000, factor=2*np.pi):
    dx = lattice_constant_tilted if tilted else lattice_constant

    dt = util.get_time_step(pathargv)

    pathA, pathB = mag_util.infer_path_B(pathA, True)
    dataA = np.loadtxt(pathA)[time_steps:]
    dataB = np.loadtxt(pathB)[time_steps:]

    if dataA.shape != dataB.shape:
        raise ValueError(f"Shape mismatch: {dataA.shape=} \t {dataB.shape=}")

    Sx = magnetization(mag_util.get_component(dataA, "x", 0),
                       mag_util.get_component(dataB, "x", 0))
    Sy = magnetization(mag_util.get_component(dataA, "y", 0),
                       mag_util.get_component(dataB, "y", 0))

    return dispersion(Sx, Sy, dx, dt, factor)




# Seems to be working
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


# TODO: Implement way to either select component you want or to give all components as dictionaries
def seebeck(dataA, dataB, eq_data, rel_step_pos):
    """

    :param dataA: mag data
    :param dataB: mag data
    :param eq_dataA: bulk data
    :param eq_dataB: bulk data
    :param rel_step_pos:
    :return:
    """
    Sz_A = mag_util.get_component(dataA, 'z', 1)
    Sz_B = mag_util.get_component(dataB, 'z', 1)

    Sz_A_eqH = bulk_util.get_components(eq_data, 'A', 'z', 1)
    Sz_B_eqH = bulk_util.get_components(eq_data, 'B', 'z', 1)

    neel = neel_vector(Sz_A, Sz_B, do_time_avg=True)
    magn = magnetization(Sz_A, Sz_B, do_time_avg=True)
    neel_eqH = neel_vector(Sz_A_eqH, Sz_B_eqH, do_time_avg=True)
    magn_eqH = magnetization(Sz_A_eqH, Sz_B_eqH, do_time_avg=True)
    neel_eqL = 1
    magn_eqL = 0

    print(f"Hot region: The equilibrium value of the neel vector is {neel_eqH}, of the magnetization is {magn_eqH}.")

    N = magn.shape[0]
    step_pos = helper.get_absolute_T_step_index(rel_step_pos, N)
    print(f"There are {N} layers. The temperature step is at {step_pos}.")

    delta_neel = np.empty_like(neel)
    delta_neel[:step_pos] = neel[:step_pos] - neel_eqH
    delta_neel[step_pos:] = neel[step_pos:] - neel_eqL

    magnon_accumulation = np.empty_like(magn)
    magnon_accumulation[:step_pos] = magn[:step_pos] - magn_eqH
    magnon_accumulation[step_pos:] = magn[step_pos:] - magn_eqL

    return magn, neel, magnon_accumulation, delta_neel

