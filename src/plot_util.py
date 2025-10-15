import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy as sp

# %%

color_by_direction = {
    (d, color) for d, color in zip(["100", "010", "110", "-110"], mpl.rcParams['axes.prop_cycle'].by_key()['color'])
}


def quick_plot_magn_neel(magnetization, neel_vector, info_string="", delta_x=0, save_suffix=None):
    x = np.arange(delta_x, neel_vector.size - delta_x, 1.0)

    fig, ax = plt.subplots()
    ax.set_xlabel("position (index)")
    ax.set_ylabel("magnitude (au)")
    ax.set_title(f"Neel (~SzA-SzB) ({info_string})")
    if delta_x > 0:
        ax.plot(x, neel_vector[delta_x:-delta_x])
    else:
        ax.plot(x, neel_vector)
    if save_suffix:
        print(f"Saving to 'out/Neel_{save_suffix}.png'")
        plt.savefig(f"out/Neel_{save_suffix}.png")
    plt.show()

    fig, ax = plt.subplots()
    ax.set_xlabel("position (index)")
    ax.set_ylabel("magnitude (au)")
    ax.set_title(f"Magn (~SzA+SzB) ({info_string})")
    if delta_x > 0:
        ax.plot(x, magnetization[delta_x:-delta_x])
    else:
        ax.plot(x, magnetization)
    if save_suffix:
        print(f"Saving to 'out/Magn_{save_suffix}.png'")
        plt.savefig(f"out/Magn_{save_suffix}.png")
    plt.show()


def quick_plot_spin_currents(j_inter_1, j_inter_2, j_intra_A, j_intra_B, j_otherpaper, info_string="", save_suffix=None):
    fig, ax = plt.subplots()
    ax.set_title(f"Spin currents ({info_string})")
    ax.set_xlabel("spin index ~ position")
    ax.set_ylabel("magnitude [au]")
    ax.plot(j_inter_1, label="j_inter_+", linewidth=0.6)
    ax.plot(j_inter_2, label="j_inter_-", linewidth=0.6)
    ax.plot(j_intra_A, label="j_intra_A", linewidth=0.6)
    ax.plot(j_intra_B, label="j_intra_B", linewidth=0.6)
    ax.plot(j_otherpaper, label="j_otherpaper", linewidth=0.6)
    ax.legend()
    plt.savefig(f"out/spin_current_{save_suffix}.png")
    plt.show()


def quick_plot_components():
    raise NotImplementedError


# formatter for multiples of pi/2
def multiple_of_pi_over_2(x, pos):
    # how many half-pi’s
    n = int(np.round(x / (np.pi / 2)))
    if n == 0:
        return r"$0$"
    elif n == 1:
        return r"$\tfrac{\pi}{2}$"
    elif n == -1:
        return r"$-\tfrac{\pi}{2}$"
    elif n == 2:
        return r"$\pi$"
    elif n == -2:
        return r"$-\pi$"
    elif n % 2 == 0:
        return fr"${n // 2}\pi$"
    else:
        return fr"${n}\tfrac{{\pi}}{{2}}$"


def place_Tstep_marking(ax, x, label=r"$\Delta T$", behind=True, **kwargs):
    plot_kwargs = dict(color="lightslategray", linestyle="--", marker="", linewidth=0.7, label=label)
    if behind:
        plot_kwargs["zorder"] = 0
    plot_kwargs.update(kwargs)

    line = ax.axvline(x, **plot_kwargs)

    return line



def add_axis_break_marking(ax, position, orientation, size=12, **kwargs):
    d = .5    # size of break diagonal

    plot_kwargs = dict(marker=[(-1., -d), (1., d)], markersize=size,
                       linestyle="none", color='k', mec='k', mew=1, clip_on=False)

    orientation = str(orientation).lower()
    if orientation in ["horizontal", "h", 1]:
        kwargs['marker'] = [(d, 1.), (-d, -1.)]
    elif orientation in ["vertical", "v", 0]:
        pass
    else:
        raise ValueError(f"Unknown orientation: {orientation}")

    for key in kwargs:
        plot_kwargs[key] = kwargs[key]

    position = str(position).lower()
    if position in ["top left", "tl", "0", "left top", "lt"]:
        x, y = 0, 1
    elif position in ["top right", "tr", "1", "right top", "rt"]:
        x, y = 1, 1
    elif position in ["bottom left", "bot left", "bl", "2", "left bottom", "left bot", "lb"]:
        x, y = 0, 0
    elif position in ["bottom right", "bot right", "br", "3", "right bottom", "right bot", "rb"]:
        x, y = 1, 0
    else:
        raise ValueError(f"Unknown position: {position}")
    ax.plot([x, ], [y, ], transform=ax.transAxes, **plot_kwargs)
    return ax




