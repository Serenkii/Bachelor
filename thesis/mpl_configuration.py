import matplotlib as mpl
import matplotlib.pyplot as plt


def configure_backends(backend="Agg", ssh=False):
    if ssh:
        print("RUNNING ON SSH")
        import os
        os.environ["DISPLAY"] = ":100"  # <-- Python way, not `export`
        print("Did you run 'xpra start :100 --daemon=yes'?")

    mpl.use(backend)
    # mpl.use('Qt5Agg')   # for interactive plots https://stackoverflow.com/questions/49844189/how-to-get-interactive-plot-of-pyplot-when-using-pycharm
    # See here: https://matplotlib.org/stable/users/explain/figure/backends.html

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
configure_backends("Qt5Agg", True)
configure()
