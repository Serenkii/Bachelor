import matplotlib as mpl
import matplotlib.pyplot as plt

def configure():
    # Use LaTeX for all text
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 300,
        # any other rcParams you like…
    })


def alt_configure():
    raise NotImplementedError()
    plt.style.use("path/to/my_mpl_style.mplstyle")

# Optionally, apply at import time:
configure()
