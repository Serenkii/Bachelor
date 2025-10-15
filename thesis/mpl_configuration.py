import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt

inches_per_pt = 1 / 72.27
tex_linewidth_pts = 404.02908
tex_linewidth_inch = tex_linewidth_pts * inches_per_pt

tex_height_pts = 693.49821
tex_height_inch = tex_height_pts * inches_per_pt


def get_tex_height(fraction=0.8):
    if fraction > 0.8:
        warnings.warn("Figure height might be very large!")
    return fraction * tex_height_inch


def inches_to_pts(inches):
    return inches / inches_per_pt


def pts_to_inches(pts):
    return pts * inches_per_pt


def get_width(fraction=1.0):
    return tex_linewidth_inch * fraction


def get_height(fraction=1.0, smaller=True):
    golden_ratio = (5 ** 0.5 - 1) / 2
    if smaller:
        return get_width(fraction) * golden_ratio
    return get_width(fraction) / golden_ratio


def get_size(fraction=1.0, fractionh=None, height_smaller=True):
    if not fractionh:
        return get_width(fraction), get_height(fraction, height_smaller)
    return get_width(fraction), get_width(fractionh)

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


def configure_for_presentation():
    latex_preamble = r"""
    \usepackage{cmbright}        % sans-serif math and text
    \usepackage{fontspec}
    \setmainfont{Arial}          % Arial for text
    \setsansfont{Arial}          % Arial for sans text
    \setmonofont{Arial}
    \usepackage{siunitx}
    \usepackage{amsmath}
    \usepackage{miller}
    \renewcommand{\familydefault}{\sfdefault}  % use sans-serif by default
    \everymath{\sf}  % ensure math is sans-serif
    \everydisplay{\sf}
    """

    mpl.rcParams.update({
        "pgf.rcfonts": False,
        "pgf.texsystem": "xelatex",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "font.size": 11,
        "figure.dpi": 300,
        "text.latex.preamble": latex_preamble,
    })

    if mpl.rcParams["figure.dpi"] < 300:
        warnings.warn("Default figure dpi is below 300.")


def configure():
    latex_preamble = (r"\usepackage{miller}"
                      r"\usepackage{siunitx}"
                      r"\usepackage{amsmath}")

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
        "lines.markersize": 6.0,
        # "lines.markersize": 3.0,
        "legend.fontsize": "small",     # default: medium
        "legend.title_fontsize": "medium",      # default is None
        "figure.dpi": 300,
        "text.latex.preamble": latex_preamble,
    })

    if mpl.rcParams["figure.dpi"] < 300:
        warnings.warn("Default figure dpi is below 300.")


def alt_configure():
    raise NotImplementedError()
    plt.style.use("path/to/my_mpl_style.mplstyle")


def default_configure():
    configure_backends("module://backend_interagg", ssh=False)  # "Qt5Agg"
    # configure_backends("PDF", ssh=False)  # "Qt5Agg"
    configure()


def presentation_configure():
    mpl.use("pgf")
    configure_for_presentation()

