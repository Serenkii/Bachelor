import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy as sp

default_slice_dict = {'t': 0, 'x': 3, 'y': 4, 'z': 5, '1': 3, '2': 4, '3': 5}

def create_slice_list(slice_string, column_dict=None):
    if not column_dict:
       column_dict = default_slice_dict
    slice_list = []
    for char in slice_string:
        if char not in column_dict.keys():
            raise ValueError(f"Can't return {slice_string}-component of spin.")
        slice_list.append(column_dict[char])
    return slice_list
