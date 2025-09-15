import os
import warnings

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy as sp



def create_slice_list(slice_string, column_dict):
    slice_list = []
    for char in slice_string:
        if char not in column_dict.keys():
            raise ValueError(f"Can't return {slice_string}-component of spin.")
        slice_list.append(column_dict[char])
    return slice_list


def get_absolute_T_step_index(relative_position, N):
    warnings.warn("Not sure what is correct...")

    func = get_index_last_warm     # In my opinion, first index cold is the correct choice!

    warnings.warn(f"Using function '{func}'")
    return func(relative_position, N)

def get_index_first_cold(relative_position, N):
    return int(np.floor(relative_position * N)) + 2

def get_index_last_warm(relative_position, N):
    return int(np.floor(relative_position * N)) + 1

def get_actual_Tstep_pos(relative_position, N):
    return get_index_last_warm(relative_position, N) + 0.5
    # TODO
