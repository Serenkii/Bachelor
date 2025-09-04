import matplotlib as mpl
import matplotlib.pyplot as plt

inches_per_pt = 1 / 72.27
tex_linewidth_pts = 404.02908
tex_linewidth_inch = tex_linewidth_pts * inches_per_pt


def get_width(fraction=1.0):
    return tex_linewidth_inch * fraction


def get_height(fraction=1.0):
    golden_ratio = (5 ** 0.5 - 1) / 2
    return get_width(fraction) * golden_ratio


def get_size(fraction=1.0):
    return get_width(fraction), get_height(fraction)


def get_frac_for_latex(width_in):
    return width_in / tex_linewidth_inch


def configure_backends(backend="Qt5Agg", ssh=False):
    if ssh:
        print("RUNNING ON SSH")
        import os
        os.environ["DISPLAY"] = ":100"  # <-- Python way, not `export`
        print("Did you run 'xpra start :100 --daemon=yes'?")

    if backend:
        mpl.use(backend)
    # mpl.use('Qt5Agg')   # for interactive plots https://stackoverflow.com/questions/49844189/how-to-get-interactive-plot-of-pyplot-when-using-pycharm
    # See here: https://matplotlib.org/stable/users/explain/figure/backends.html

def configure():
    latex_preamble = r"\usepackage{miller}"

    # Use LaTeX for all text
    mpl.rcParams.update({
        "text.usetex": True,
        # "figure.figsize": [6.4, 4.8],       # default
        "figure.figsize": list(get_size()),
        "font.family": "serif",
        "font.serif": ["Palatino"],
        "font.size": 11,
        # "axes.labelsize": 14,
        # "xtick.labelsize": 12,
        # "ytick.labelsize": 12,
        # "legend.fontsize": 12,
        "figure.dpi": 300,
        "text.latex.preamble": latex_preamble,
    })


def alt_configure():
    raise NotImplementedError()
    plt.style.use("path/to/my_mpl_style.mplstyle")


def default_configure():
    configure_backends("module://backend_interagg", ssh=False)  # "Qt5Agg"
    configure()



