import os

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
