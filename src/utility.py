import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy as sp



def plot_spin_xyz_over_t(file_path, title=""):
    data = np.loadtxt(file_path)
    fig, ax = plt.subplots()
    ax.title.set_text(title)
    ax.set_xlabel("time in ps")
    ax.set_ylabel("y")
    ax.plot(data[:, 0], data[:, 3], label="sx")
    ax.plot(data[:, 0], data[:, 4], label="sy")
    ax.plot(data[:, 0], data[:, 5], label="sz")
    ax.legend()

    plt.show()



