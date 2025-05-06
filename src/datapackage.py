import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy as sp

import bulk_util
import mag_util


class DataPackage:
    def __init__(self, txt_source_file=None, npy_source_file=None, data=None, description=""):
        # TODO
        self.txt_source_file = txt_source_file
        self.npy_source_file = npy_source_file
        self.data = data
        self.description = description

    def __str__(self):
        return f"{super().__str__()} | {self.description}"


class BulkDataPackage(DataPackage):
    # TODO

    def __init__(self, txt_source_file=None, npy_source_file=None, data=None, description=""):
        super().__init__(txt_source_file=None, npy_source_file=None, data=None, description="")



class MagneticProfileDataPackage(DataPackage):
    # TODO

    def __init__(self, txt_source_file=None, npy_source_file=None, data=None, description=""):
        super().__init__(txt_source_file=None, npy_source_file=None, data=None, description="")

