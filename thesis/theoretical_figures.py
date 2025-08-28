import numpy as np
import thesis.mpl_configuration

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import ImageGrid

# %%

seeblau = "#00a9e0"
font_size = 12

# %%

from PIL import Image, ImageChops, ImageEnhance

import fitz


def crop_pdf_to_content(input_pdf, output_pdf, dpi=1500, margin_x0=0, margin_y0=0, margin_x1=0, margin_y1=0):
    print(f"Trying to crop pdf {input_pdf}...", end="\t")

    doc = fitz.open(input_pdf)

    for page in doc:
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Increase contrast to make faint lines more visible
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(10.0)  # You can try 3.0, 5.0, or higher

        # Convert white to transparency for better bbox detection
        bg = Image.new("RGB", img.size, (255, 255, 255))
        diff = ImageChops.difference(img, bg)
        bbox = diff.getbbox()

        if bbox:
            x0, y0, x1, y1 = bbox
            x0 = max(x0 - margin_x0, 0)
            y0 = max(y0 - margin_y0, 0)
            x1 = min(x1 + margin_x1, img.width)
            y1 = min(y1 + margin_y1, img.height)
            scale_x = page.rect.width / pix.width
            scale_y = page.rect.height / pix.height

            crop_rect = fitz.Rect(
                x0 * scale_x,
                (pix.height - y1) * scale_y,
                x1 * scale_x,
                (pix.height - y0) * scale_y
            )

            page.set_cropbox(crop_rect)

    print(f"Saving into {output_pdf}...")
    doc.save(output_pdf)


# %% draw a vector / fancy arrow
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import ArrowStyle
from mpl_toolkits.mplot3d import proj3d
import matplotlib.patches as patches
from matplotlib.path import Path


class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    @classmethod
    def from_midpoint_and_direction(cls, midpoint, direction, *args, length_arrow=None, **kwargs):
        midpoint = np.array(midpoint)
        direction = np.array(direction)
        if length_arrow:
            norm = np.sqrt(np.dot(direction, direction))
            direction *= length_arrow / norm
        half_vec = direction * 0.5
        start = midpoint - half_vec
        end = midpoint + half_vec
        xs, ys, zs = [start[0], end[0]], [start[1], end[1]], [start[2], end[2]]
        return cls(xs, ys, zs, *args, **kwargs)

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        proj = self.axes.get_proj()  # <- use projection matrix from Axes3D
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, proj)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        proj = self.axes.get_proj()
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, proj)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        print("WARNING: Method created with no clue what I am doing.")
        return min(np.min(xs), np.min(ys), np.min(zs)) - 1e3  # Used for depth sorting
        # TODO: I have no idea if this makes sense


# %% LLG
def llg_equation(save_name=None, thermal_noise=0.0):
    font_size = 12

    A = 0.3
    dA0 = +0.2
    w = 5
    tau = 0.6
    H0 = np.array([0.0, 0.0, 1.0])

    gamma = 0.5
    alpha = 0.1
    mu_s = 0.3
    frac = gamma / ((1 + alpha ** 2) * mu_s)

    thermal_noise_amplitude = thermal_noise  # Change this value to get stochastic

    def a(t):
        return dA0 * np.exp(-t / tau) + A

    def Sx(t):
        return a(t) * np.cos(w * t)

    def Sy(t):
        return a(t) * np.sin(w * t)

    def Sz(t):
        return np.sqrt(1 - a(t) ** 2)

    def S(t):
        return np.array([Sx(t), Sy(t), Sz(t)])

    def thermal_noise(t, i=None):
        ran = np.random.default_rng(i)
        return ran.normal(0, thermal_noise_amplitude, 3)

    def H(t, i):
        return H0 + thermal_noise(t, i)

    def damping(S, H):
        return - frac * alpha * np.cross(S, np.cross(S, H))

    def gilbert_torque(S, H):
        return - frac * np.cross(S, H)

    def dS(S, H, dt):
        return dt * (damping(S, H) + gilbert_torque(S, H))

    def S_llg(S0, t_arr):
        S = np.empty(shape=(t_arr.shape[0], 3))
        S[0] = S0
        for i in range(1, t_arr.size):
            S[i] = S[i - 1] + dS(S[i - 1], H(t_arr[i - 1], i), t_arr[i] - t_arr[i - 1])
        return S

    t = np.linspace(-0.4, 20, 20000)

    fig, ax = plt.subplots()
    ax.plot(t, Sx(t), label="Sx")
    ax.plot(t, Sy(t), label="Sy")
    ax.plot(t, Sz(t), label="Sz")
    ax.plot(t, a(t), label="a")

    ax.legend()
    plt.show()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    u, v = np.mgrid[0:2 * np.pi:20j, 0:(np.pi / 2):10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="grey", alpha=0.5, linewidth=0.1)

    # ax.plot(Sx(t), Sy(t), Sz(t), linestyle="--")

    S_L = S_llg(S(0), t)
    # S_L = S_llg(np.array([0.0, 0.0, 1.0]), t)
    ax.plot(S_L[:, 0], S_L[:, 1], S_L[:, 2], linestyle="--", linewidth=0.8, color="#00a9e0")

    t0 = -0.2
    i_t = np.argmin(np.abs(t - t0))

    H0_arrow = Arrow3D([- 0.0 * H0[0], 1.2 * H0[0]], [- 0.0 * H0[1], 1.2 * H0[1]], [- 0.0 * H0[2], 1.2 * H0[2]],
                       mutation_scale=20, shrinkA=0, shrinkB=0, lw=1, arrowstyle="-|>", color="k", )
    ax.add_artist(H0_arrow)
    ax.text(1.1 * H0[0] - 0.08, 1.1 * H0[1], 1.1 * H0[2] + 0.02, r'$\vec{H}_i$', size=font_size, ha="right", va="top")

    if thermal_noise_amplitude != 0.0:
        H_ = H(t0, i_t)
        H_arrow = Arrow3D([- 0.0 * H_[0], 1.2 * H_[0]], [- 0.0 * H_[1], 1.2 * H_[1]], [- 0.0 * H_[2], 1.2 * H_[2]],
                          mutation_scale=20,
                          shrinkA=0, shrinkB=0, lw=1, arrowstyle="-|>", color="k", )
        ax.add_artist(H_arrow)

    # S_arrow = Arrow3D([0, Sx(t0)], [0, Sy(t0)], [0, Sz(t0)], mutation_scale=20,
    #                   shrinkA=0, shrinkB=0, lw=1, arrowstyle="-|>", color="k",)
    S_arrow = Arrow3D([0, S_L[i_t, 0]], [0, S_L[i_t, 1]], [0, S_L[i_t, 2]], mutation_scale=20,
                      shrinkA=0, shrinkB=0, lw=1, arrowstyle="-|>", color="k", )
    ax.add_artist(S_arrow)
    ax.text(*(S_L[i_t] - np.array([0.02, 0.02, 0.1])), r'$\vec{S}_i$', size=font_size, ha="left", va="top")

    # damping_vec = damping(S(t0), H(t0))
    damping_vec = damping(S_L[i_t], H(t0, i_t))
    damping_vec *= 0.3 / np.sqrt(damping_vec.dot(damping_vec))
    # damping_vec += S(t0)
    damping_vec += S_L[i_t]
    # damping_arrow = Arrow3D([Sx(t0), damping_vec[0]], [Sy(t0), damping_vec[1]], [Sz(t0), damping_vec[2]],
    #                         mutation_scale=20, shrinkA=0, shrinkB=0, lw=1, arrowstyle="-|>", color="k")
    damping_arrow = Arrow3D([S_L[i_t, 0], damping_vec[0]], [S_L[i_t, 1], damping_vec[1]], [S_L[i_t, 2], damping_vec[2]],
                            mutation_scale=20, shrinkA=0, shrinkB=0, lw=1, arrowstyle="-|>", color="#53c412")
    ax.add_artist(damping_arrow)
    ax.text(*(damping_vec + np.array([0.07, 0.02, -0.09])),
            r'$- \vec{S}_i \times \left(\vec{S}_i \times \vec{H}_i \right)$',
            size=font_size, ha="right", va="top", color="#53c412")  # backgroundcolor=(1.0, 1.0, 1.0, 0.6)

    # gilbert_vec = gilbert_torque(S(t0), H(t0))
    gilbert_vec = gilbert_torque(S_L[i_t], H(t0, i_t))
    gilbert_vec *= 0.3 / np.sqrt(gilbert_vec.dot(gilbert_vec))
    # gilbert_vec += S(t0)
    gilbert_vec += S_L[i_t]
    # gilbert_arrow = Arrow3D([Sx(t0), gilbert_vec[0]], [Sy(t0), gilbert_vec[1]], [Sz(t0), gilbert_vec[2]],
    #                         mutation_scale=20, shrinkA=0, shrinkB=0, lw=1, arrowstyle="-|>", color="k")
    gilbert_arrow = Arrow3D([S_L[i_t, 0], gilbert_vec[0]], [S_L[i_t, 1], gilbert_vec[1]], [S_L[i_t, 2], gilbert_vec[2]],
                            mutation_scale=20, shrinkA=0, shrinkB=0, lw=1, arrowstyle="-|>", color="#f47c20")
    ax.add_artist(gilbert_arrow)
    ax.text(*(gilbert_vec - np.array([0.05, 0.05, 0.08])), r'$- \vec{S}_i \times \vec{H}_i$',
            size=font_size, ha="left", va="top", color="#f47c20")

    ax.view_init(elev=20., azim=-15, roll=0)

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(0.0, 1.0)
    ax.set_box_aspect((1.0, 1.0, 1.0), zoom=4)
    ax.set_aspect("equal")

    ax.set_axis_off()

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)  # No border padding

    if save_name:
        fig.savefig(save_name, bbox_inches='tight', pad_inches=0)

        crop_pdf_to_content(save_name,
                            f"{save_name[:-4]}_cropped.pdf",
                            margin_y0=400, margin_y1=-100, margin_x0=100, margin_x1=100)

    plt.show()


# %% DMI
def vector_in_plane_2d(axis, position: tuple, radius=0.2, color="black", **kwargs):
    circle_in = patches.Circle(position, radius, fill=False, edgecolor=color, **kwargs)
    axis.add_patch(circle_in)
    diag_half = np.sqrt(2) * 0.2 * 0.5  # = 1/np.sqrt(2) * 0.2
    axis.plot([position[0] - diag_half, position[0] + diag_half], [position[1] - diag_half, position[1] + diag_half],
              color=color, **kwargs)  # diagonal \
    axis.plot([position[0] - diag_half, position[0] + diag_half], [position[1] + diag_half, position[1] - diag_half],
              color=color, **kwargs)  # diagonal /


def vector_out_plane_2d(axis, position: tuple, radius=0.2, radius_dot=None, color="black", **kwargs):
    radius_dot = radius_dot or radius * 0.25
    circle_out = patches.Circle(position, radius, fill=False, edgecolor=color, **kwargs)
    dot = patches.Circle(position, radius_dot, color=color, **kwargs)
    axis.add_patch(circle_out)
    axis.add_patch(dot)


def arrow_2d_from_midpoint(midpoint, direction, *args, length_arrow=None, **kwargs):
    midpoint = np.array(midpoint)
    direction = np.array(direction)
    if length_arrow:
        norm = np.sqrt(np.dot(direction, direction))
        direction *= length_arrow / norm
    half_vec = direction * 0.5
    start = midpoint - half_vec
    end = midpoint + half_vec
    return FancyArrowPatch(start, end, *args, **kwargs)


def dmi1(save_name=None):
    font_size = 14

    b = 0.0
    D = 0.5
    S = 1
    S1 = np.array([0, b, S])
    S2 = np.array([0, b, -S])
    D_vec = np.array([D, 0, 0])

    print(np.dot(-D_vec, np.cross(S1, S2)))

    fig, ax = plt.subplots()

    # connecting lines
    ax.plot([-1, 1], [0, 0], marker="", linestyle="-", linewidth=0.5, color="k")

    # Spin 1
    S1_arrow = arrow_2d_from_midpoint([-1, 0], S1[1:], shrinkA=0, shrinkB=0, mutation_scale=30, lw=3, arrowstyle="-|>",
                                      color="k")
    ax.add_artist(S1_arrow)
    ax.plot(-1, 0, linestyle="", marker="o", markersize=10, color=seeblau)
    ax.text(-1.1, 0, r"$\vec{S}_i$", size=font_size, color=seeblau, ha="right", va="bottom")

    # Spin 2
    S2_arrow = arrow_2d_from_midpoint([1, 0], S2[1:], shrinkA=0, shrinkB=0, mutation_scale=30, lw=3, arrowstyle="-|>",
                                      color="k")
    ax.add_artist(S2_arrow)
    ax.plot(1, 0, linestyle="", marker="o", markersize=10, color=seeblau)
    ax.text(1.1, 0, r"$\vec{S}_j$", size=font_size, color=seeblau, ha="left", va="bottom")

    # red atom
    ax.plot(0, 0, marker="p", linestyle="", markersize=15, color="r")

    # symmetry point
    ax.plot(0, 0, marker=".", linestyle="", color="green")
    ax.text(0.1, 0.1, r"$I$", size=font_size, color="green", ha="left", va="bottom")

    # coordinate system
    pos = (1.3, -0.5)
    length = 0.3
    vector_out_plane_2d(ax, (1.3, -0.5), 0.1)
    y_arrow = FancyArrowPatch(pos, (pos[0] + length, pos[1]),
                              shrinkA=0, shrinkB=0, mutation_scale=10, lw=1, arrowstyle="-|>", color="k")
    z_arrow = FancyArrowPatch(pos, (pos[0], pos[1] + length),
                              shrinkA=0, shrinkB=0, mutation_scale=10, lw=1, arrowstyle="-|>", color="k")
    ax.add_artist(y_arrow)
    ax.add_artist(z_arrow)
    ax.text(pos[0] - 0.07, pos[1] - 0.07, "$x$", size=font_size, ha="right", va="top")
    ax.text(pos[0] + 0.2, pos[1] - 0.05, "$y$", size=font_size, ha="left", va="top")
    ax.text(pos[0] + 0.05, pos[1] + 0.2, "$z$", size=font_size, ha="left", va="bottom")

    ax.set_xlim(-2, 2)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')

    ax.set_axis_off()

    if save_name:
        fig.savefig(save_name, bbox_inches='tight', pad_inches=0)

    plt.show()


def dmi2(save_name=None):
    font_size = 14

    b = 0.1
    D = 0.5
    S = 1
    S1 = np.array([0, b, S])
    S2 = np.array([0, b, -S])
    D_vec = np.array([D, 0, 0])

    print(np.dot(-D_vec, np.cross(S1, S2)))

    fig, ax = plt.subplots()

    # connecting lines
    ax.plot([-1, 1], [0, 0], marker="", linestyle="-", linewidth=0.5, color="k")
    ax.plot([-1, 0, 1], [0, 0.45, 0], marker="", color="grey", linewidth=0.3)

    # Spin 1
    S1_arrow = arrow_2d_from_midpoint([-1, 0], S1[1:], shrinkA=0, shrinkB=0, mutation_scale=30, lw=3, arrowstyle="-|>",
                                      color="k")
    ax.add_artist(S1_arrow)
    ax.plot(-1, 0, linestyle="", marker="o", markersize=10, color=seeblau)
    ax.text(-1.1, 0, r"$\vec{S}_i$", size=font_size, color=seeblau, ha="right", va="bottom")

    # Spin 2
    S2_arrow = arrow_2d_from_midpoint([1, 0], S2[1:], shrinkA=0, shrinkB=0, mutation_scale=30, lw=3, arrowstyle="-|>",
                                      color="k")
    ax.add_artist(S2_arrow)
    ax.plot(1, 0, linestyle="", marker="o", markersize=10, color=seeblau)
    ax.text(1.1, 0, r"$\vec{S}_j$", size=font_size, color=seeblau, ha="left", va="bottom")

    # symmetry axis
    ax.plot([0, 0], [-0.8, 0.8], marker="", linestyle="-.", color="green")
    ax.text(0.4, -0.6, r"$C_2$", size=font_size, color="green", ha="left", va="top")

    # red atom
    ax.plot(0, 0.45, marker="p", linestyle="", markersize=15, color="r")

    # dmi vector
    vector_out_plane_2d(ax, (0, 0), color="orange", zorder=5)
    ax.text(0.15, -0.15, r"$\vec{D}_{ij}$", size=font_size, color="orange", ha="left", va="top")

    # Add curved arrow (rotation indicator)
    center = (0, -0.65)
    width = 0.6
    height = 0.2
    theta2 = 50  # ending angle in degrees

    arc = patches.Arc(center, width, height, angle=0, theta1=130, theta2=theta2, lw=0.8, color='black', zorder=5)
    ax.add_patch(arc)

    # Convert angle to radians
    angle_rad = np.radians(theta2)

    # Compute the endpoint of the arc
    x_end = center[0] + (width / 2) * np.cos(angle_rad)
    y_end = center[1] + (height / 2) * np.sin(angle_rad)

    # Compute a nearby point slightly before the end (for direction)
    angle_back = np.radians(theta2 - 5)
    x_back = center[0] + (width / 2) * np.cos(angle_back)
    y_back = center[1] + (height / 2) * np.sin(angle_back)

    # Add arrowhead
    ax.annotate("",
                xy=(x_end, y_end),
                xytext=(x_back, y_back),
                arrowprops=dict(arrowstyle="-|>", color='black', lw=0.8))

    ax.set_xlim(-2, 2)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')

    ax.set_axis_off()

    if save_name:
        fig.savefig(save_name, bbox_inches='tight', pad_inches=0)

    plt.show()


# %% Spin waves
def spin_waves(save_name):
    N = 8
    phi0 = 0
    n = np.arange(N)
    amplitude = 0.3
    a = 1
    k = 2 * np.pi / (N * a)
    phi = phi0 + k * n * a
    plane_component = amplitude * np.exp(- 1j * phi)
    Sx = np.real(plane_component)
    Sy = - np.imag(plane_component)
    Sz = np.sqrt(1 - amplitude ** 2)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    for l in n:
        spin_arrow = Arrow3D([l, l + Sx[l]], [0, Sy[l]], [0, Sz], mutation_scale=20,
                             shrinkA=0, shrinkB=0, lw=1, arrowstyle="-|>", color="k", zorder=0)
        ax.add_artist(spin_arrow)

    n_continuous = np.linspace(Sx[0] - 0.8, N - 1 + Sx[-1] + 0.2, 100)
    phi_continuous = phi0 + k * n_continuous * a
    Sy_continuous = - np.imag(amplitude * np.exp(- 1j * phi_continuous))
    ax.plot(n_continuous, Sy_continuous, zs=Sz, zdir='z', linestyle="--", marker="", color=seeblau, linewidth=1)

    angle = np.linspace(0, 2 * np.pi)
    circle_x = amplitude * np.cos(angle)
    circle_y = amplitude * np.sin(angle)
    for l in n:
        ax.plot(circle_x + l, circle_y, zs=Sz, zdir='z', linestyle="-", marker="", color=seeblau, linewidth=1)

    ax.plot(np.linspace(Sx[0] - 0.8, N - 1 + Sx[-1] + 0.2, 10), 0.0, zs=0, zdir='z', marker="", linestyle=":",
            color="k", linewidth=0.5)
    ax.plot(n, 0.0, zs=0, zdir='z', marker="o", linestyle="", color="k", markersize=2)

    ax.view_init(elev=30., azim=-100, roll=0)

    ax.set_xlim(Sx[0] - 0.8 - 0.2, N - 1 + Sx[-1] + 0.2 + 0.2)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0.0, 1.0)
    ax.set_box_aspect((1.0, 1.0, 1.0), zoom=4)
    ax.set_aspect("equal")

    ax.set_axis_off()

    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)  # No border padding

    if save_name:
        fig.savefig(save_name, bbox_inches='tight', pad_inches=0)

        crop_pdf_to_content(save_name,
                            f"{save_name[:-4]}_cropped.pdf",
                            margin_y0=-150, margin_y1=300, margin_x0=50, margin_x1=150)

    plt.show()


# %% AFM modes

def afm_modes(save_name=None):
    font_size = 12

    H_E = 2  # exchange
    H_A = 0.2  # anisotropy
    H_C = np.sqrt(2 * H_E * H_A + H_A * H_A)

    S1_over_S2_a = - ((H_E + H_A) + H_C) / H_E
    S1_over_S2_b = - ((H_E + H_A) - H_C) / H_E

    Sx_2_a = 0.2
    Sy_2_a = 0.0
    Sz_2_a = - np.sqrt(1 - Sx_2_a ** 2 - Sy_2_a ** 2)
    Sx_1_a = S1_over_S2_a * Sx_2_a
    Sy_1_a = 0.0
    Sz_1_a = np.sqrt(1 - Sx_1_a ** 2 - Sy_1_a ** 2)

    Sx_1_b = 0.2
    Sy_1_b = 0.0
    Sz_1_b = np.sqrt(1 - Sx_2_a ** 2 - Sy_2_a ** 2)
    Sx_2_b = Sx_1_b / S1_over_S2_b
    Sy_2_b = 0.0
    Sz_2_b = - np.sqrt(1 - Sx_2_b ** 2 - Sy_2_b ** 2)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # spin arrows
    arrow_S2_a = Arrow3D([-1, -1 + Sx_2_a], [0, Sy_2_a], [0, Sz_2_a], mutation_scale=20,
                         shrinkA=0, shrinkB=0, lw=1, arrowstyle="-|>", color="r", zorder=0)
    ax.add_artist(arrow_S2_a)
    arrow_S1_a = Arrow3D([-1, -1 + Sx_1_a], [0, Sy_1_a], [0, Sz_1_a], mutation_scale=20,
                         shrinkA=0, shrinkB=0, lw=1, arrowstyle="-|>", color="b", zorder=0)
    ax.add_artist(arrow_S1_a)

    arrow_S2_b = Arrow3D([1, 1 + Sx_2_b], [0, Sy_2_b], [0, Sz_2_b], mutation_scale=20,
                         shrinkA=0, shrinkB=0, lw=1, arrowstyle="-|>", color="r", zorder=0)
    ax.add_artist(arrow_S2_b)
    arrow_S1_b = Arrow3D([1, 1 + Sx_1_b], [0, Sy_1_b], [0, Sz_1_b], mutation_scale=20,
                         shrinkA=0, shrinkB=0, lw=1, arrowstyle="-|>", color="b", zorder=0)
    ax.add_artist(arrow_S1_b)

    # precession circles
    phi = np.linspace(0, 2 * np.pi)
    circle_x = np.cos(phi)
    circle_y = np.sin(phi)

    circle_kwargs = dict(linestyle="--", linewidth=0.8)
    ax.plot(Sx_1_a * circle_x - 1, Sx_1_a * circle_y, Sz_1_a, **circle_kwargs, color="b")
    ax.plot(Sx_2_a * circle_x - 1, Sx_2_a * circle_y, Sz_2_a, **circle_kwargs, color="r")
    ax.plot(Sx_1_b * circle_x + 1, Sx_1_b * circle_y, Sz_1_b, **circle_kwargs, color="b")
    ax.plot(Sx_2_b * circle_x + 1, Sx_2_b * circle_y, Sz_2_b, **circle_kwargs, color="r")

    # precession direction arrow
    def draw_direction_arrow(x_pos, chirality, radius=0.3):
        # curved arrow for precession direction
        # define an arc from angle theta1 to theta2
        theta = np.linspace(np.pi / 1.5 + 0.2, 2 * np.pi + np.pi / 3 - 0.2, 30)  # arc from 0 to ~120 degrees
        theta = theta[::chirality]
        x_arc = radius * np.cos(theta) + x_pos  # centered at x = x_pos
        y_arc = radius * np.sin(theta)
        z_arc = np.zeros_like(theta)

        # draw arc
        ax.plot(x_arc, y_arc, z_arc, color="k", linestyle="-", linewidth=1, zorder=100)

        # compute direction of arrowhead (tangent to arc at the end)
        dx = -radius * np.sin(theta[-1]) * chirality
        dy = radius * np.cos(theta[-1]) * chirality
        dz = 0

        arrowhead = Arrow3D.from_midpoint_and_direction(
            midpoint=[x_arc[-1], y_arc[-1], 0],  # move back to avoid overshooting
            direction=[dx, dy, dz],
            mutation_scale=15,
            lw=0.0,
            arrowstyle="-|>",
            color="k",
            length_arrow=0.3
        )
        ax.add_artist(arrowhead)

    draw_direction_arrow(-1, 1, radius=0.35)
    draw_direction_arrow(1, -1, radius=0.35)

    # text labels
    ax.text(-1, 0, 1.1, r"$\omega_{\alpha} > 0$", size=font_size, va="bottom", ha="center")
    ax.text(1, 0, 1.1, r"$\omega_{\beta} < 0$", size=font_size, va="bottom", ha="center")

    ax.text(Sx_1_a - 1, 0, 0.5 * Sz_1_a, r"$\vec{S}_1$", size=font_size, va="center", ha="right", color="b")
    ax.text(Sx_2_a - 1, 0, 0.5 * Sz_2_a, r"$\vec{S}_2$", size=font_size, va="center", ha="left", color="r")
    ax.text(Sx_1_b + 1, 0, 0.5 * Sz_1_b, r"$\vec{S}_1$", size=font_size, va="center", ha="left", color="b")
    ax.text(Sx_2_b + 1, 0, 0.5 * Sz_2_b, r"$\vec{S}_2$", size=font_size, va="center", ha="right", color="r")

    # z axis
    ax.plot([-1.0, -1.0], [0.0, 0.0], [-1.0, 1.0], marker="", linestyle="--", color="k", linewidth=0.8)
    ax.plot([1.0, 1.0], [0.0, 0.0], [-1.0, 1.0], marker="", linestyle="--", color="k", linewidth=0.8)
    z_axis_arrow = Arrow3D([0.0, 0.0], [0, 0.0], [0.2, 0.8], mutation_scale=10,
                           shrinkA=0, shrinkB=0, lw=1, arrowstyle="-|>", color="k", zorder=0)
    ax.add_artist(z_axis_arrow)
    ax.text(0.05, 0.0, 0.8, r"$z$", size=font_size, va="top", ha="left")

    # view specs
    ax.view_init(elev=20., azim=-90, roll=0)

    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.5)
    ax.set_box_aspect((1.0, 1.0, 1.0), zoom=4)
    ax.set_aspect("equal")

    ax.set_axis_off()

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)  # No border padding

    if save_name:
        fig.savefig(save_name, bbox_inches='tight', pad_inches=0)

        crop_pdf_to_content(save_name,
                            f"{save_name[:-4]}_cropped.pdf",
                            margin_y0=250, margin_y1=0, margin_x0=100, margin_x1=150)

    plt.show()


# %% Toy Model

def toy_model(save_name=None):
    colors = ["red", "blue"]

    fig, ax = plt.subplots()
    N = 6
    corner_dN = 2

    def draw_points():
        for y in range(N):
            for x in range(N):
                if x >= N - corner_dN and y >= N - corner_dN:
                    continue
                color = colors[(x + y) % 2]
                ax.plot(x, -y, color=color, linestyle="", marker="o", ms=8)

    def draw_axes(position, box_width, distance_txt=0.05, text_size=10):
        position = np.array(position)
        arrow_kwargs = dict(shrinkA=0, shrinkB=0, mutation_scale=6, lw=1, arrowstyle="<|-|>", color="k")

        previous_font_size = mpl.rcParams["font.size"]
        mpl.rcParams["font.size"] = text_size

        start = position - np.array([box_width / 2, 0])
        end = position + np.array([box_width / 2, 0])
        arrow = FancyArrowPatch(start, end, **arrow_kwargs)
        ax.add_patch(arrow)
        ax.text(end[0], end[1] - distance_txt, r"$x$", va="top", ha="right")
        ax.text(end[0], end[1] + distance_txt, r"$\hkl[100]$", va="bottom", ha="center")
        ax.text(start[0], start[1] + distance_txt, r"$\hkl[-100]$", va="bottom", ha="center")

        start = position - np.array([0, box_width / 2])
        end = position + np.array([0, box_width / 2])
        arrow = FancyArrowPatch(start, end, **arrow_kwargs)
        ax.add_patch(arrow)
        ax.text(end[0] - distance_txt * 2, end[1], r"$y$", va="top", ha="right")
        ax.text(end[0], end[1] + distance_txt, r"$\hkl[010]$", va="bottom", ha="center")
        ax.text(start[0], start[1] - distance_txt, r"$\hkl[0-10]$", va="top", ha="center")

        start = position - np.array([box_width / 2, box_width / 2])
        end = position + np.array([box_width / 2, box_width / 2])
        arrow = FancyArrowPatch(start, end, **arrow_kwargs)
        ax.add_patch(arrow)
        ax.text(start[0], start[1] - distance_txt, r"$\hkl[-1-10]$", va="top", ha="center")
        ax.text(end[0], end[1] + distance_txt, r"$\hkl[110]$", va="bottom", ha="center")

        start = position + np.array([- box_width / 2, box_width / 2])
        end = position + np.array([box_width / 2, - box_width / 2])
        arrow = FancyArrowPatch(start, end, **arrow_kwargs)
        ax.add_patch(arrow)
        ax.text(start[0], start[1] + distance_txt, r"$\hkl[-110]$", va="bottom", ha="center")
        ax.text(end[0], end[1] - distance_txt, r"$\hkl[1-10]$", va="top", ha="center")

        mpl.rcParams["font.size"] = previous_font_size

    def draw_lattice_constant(distance=0.2):
        arrow_kwargs = dict(shrinkA=0, shrinkB=0, mutation_scale=3, lw=1, arrowstyle="|-|", color="k")

        bar = FancyArrowPatch((0 - distance, -4), (0 - distance, -5), **arrow_kwargs)
        ax.add_patch(bar)
        bar = FancyArrowPatch((0, -5 - distance), (1, -5 - distance), **arrow_kwargs)
        ax.add_patch(bar)

        ax.text(0 - distance - 0.1, -4.5, r"$a$", size=font_size, va="center", ha="right")
        ax.text(0.5, -5 - distance - 0.1, r"$a$", size=font_size, va="top", ha="center")

    def draw_wigner_seitz(color="grey"):
        ds = 0.3
        ax.plot([0, 1, 2, 1, 0], [-4, -3, -4, -5, -4], marker="", linestyle="-", linewidth=0.5, color=color)
        ax.plot([1, 2, 3, 2], [-3, -2, -3, -4], marker="", linestyle="--", linewidth=0.5, color=color)
        ax.plot([2, 3, 3 - ds], [-4, -5, -5 - ds], marker="", linestyle="--", linewidth=0.5, color=color)
        ax.plot([1, 1 + ds], [-5, -5 - ds], marker="", linestyle="--", linewidth=0.5, color=color)

        ax.text(-0.2, -3.2, "Wigner-Seitz\ncell", va="top", ha="center", color=color)

        ax.fill([0, 1, 2, 1, 0], [-4, -3, -4, -5, -4], color=color, alpha=0.3)

    def draw_lattice_vectors(color1="black", color2="darkgoldenrod"):
        arrow_kwargs = dict(shrinkA=0, shrinkB=0, mutation_scale=15, lw=1, arrowstyle="-|>")

        start = np.array([1, -4])
        end = np.array([2, -3])
        arrow = FancyArrowPatch(start, end, **arrow_kwargs, color=color1, zorder=5)
        ax.add_patch(arrow)
        position_text = start + 0.7 * (end - start)
        ax.text(position_text[0], position_text[1], r"$\vec{a}_1$", color=color1, va="bottom", ha="right")

        end = [2, -5]
        arrow = FancyArrowPatch(start, end, **arrow_kwargs, color=color1, zorder=5)
        ax.add_patch(arrow)
        position_text = start + 0.7 * (end - start)
        ax.text(position_text[0], position_text[1], r"$\vec{a}_2$", color=color1, va="top", ha="right")

        end = [2, -4]
        arrow = FancyArrowPatch(start, end, **arrow_kwargs, color=color2, zorder=5)
        ax.add_patch(arrow)
        position_text = start + 0.65 * (end - start)
        ax.text(position_text[0], position_text[1], r"$\vec{r}_{\mathrm{B}}$", color=color2, va="bottom", ha="center")

        ax.plot(start[0], start[1], color=color2, marker='o', ms=2, zorder=5)
        ax.text(start[0] - 0.055, start[1] + 0.055, r"$\vec{r}_{\mathrm{A}}$", color=color2, va="bottom", ha="right")

    def draw_exchange_interactions(color_J1="green", color_J2_1="magenta", color_J2_2="purple", text_size=10):
        shared_kwargs = dict(marker="", linestyle="-")
        paths_x = dict(
            J1=[[4, 4], [3, 5],
                [3, 3], [2, 3]],
            J2_1=[[3, 5], [2, 4]],
            J2_2=[[3, 5], [3, 5]]
        )
        paths_y = dict(
            J1=[[0, -2], [-1, -1],
                [0, -2], [-1, -1]],
            J2_1=[[0, -2], [-2, 0]],
            J2_2=[[-2, 0], [-1, -3]]
        )
        paths_kwargs = dict(
            J1=[dict(color=color_J1, lw=2), dict(color=color_J1, lw=2),
                dict(color=color_J1, lw=1), dict(color=color_J1, lw=1)],
            J2_1=[dict(color=color_J2_1, lw=1.8), dict(color=color_J2_1, lw=1)],
            J2_2=[dict(color=color_J2_2, lw=1.8), dict(color=color_J2_2, lw=1)]
        )

        for J in paths_x:
            for path_x, path_y, path_kwargs in zip(paths_x[J], paths_y[J], paths_kwargs[J]):
                ax.plot(path_x, path_y, **path_kwargs, **shared_kwargs)

        # Text
        previous_font_size = mpl.rcParams["font.size"]
        mpl.rcParams["font.size"] = text_size

        d = 0.06
        d_ = 0.5 * np.sqrt(2) * d

        ax.text(4.5, -1 - d, r"$J_1$", va="top", ha="center", color=color_J1)
        ax.text(4 + d, -0.5, r"$J_1$", va="center", ha="left", color=color_J1)
        ax.text(3.5, -1 + d, r"$J_1$", va="bottom", ha="center", color=color_J1)
        ax.text(4 - d, -1.5, r"$J_1$", va="center", ha="right", color=color_J1)

        ax.text(4.75 + d_, -0.25 - d_, r"$J_2'$", va="top", ha="left", color=color_J2_2)
        ax.text(3.25 + d_, -1.75 - d_, r"$J_2'$", va="top", ha="left", color=color_J2_2)

        ax.text(4.75 - d_, -1.75 - d_, r"$J_2$", va="top", ha="right", color=color_J2_1)
        ax.text(3.25 + d_, -0.25 + d_, r"$J_2$", va="bottom", ha="left", color=color_J2_1)

        mpl.rcParams["font.size"] = previous_font_size

    def add_circular_arrow(center, radius, theta1, theta2,
                           color='black', lw=0.7, delta_angle=6, mutation_scale=7, zorder=5):
        """
        Draw a circular arc (part of a circle) and add an arrowhead at the end.
        - center: (cx, cy)
        - radius: circle radius
        - theta1, theta2: start and end angles in degrees (measured ccw from +x)
        - delta_angle: degrees back from theta2 to compute arrow tail (controls arrow direction)
        """
        cx, cy = center
        # draw arc (circle -> width == height == 2*radius)
        arc = patches.Arc(center, 2 * radius, 2 * radius, angle=0, theta1=theta1, theta2=theta2,
                          lw=lw, color=color)
        ax.add_patch(arc)

        # helper to get point on circle at angle t (degrees)
        def pt(tdeg):
            t = np.radians(tdeg)
            return cx + radius * np.cos(t), cy + radius * np.sin(t)

        x_end, y_end = pt(theta2)
        x_back, y_back = pt(theta2 - delta_angle)

        # arrowhead from a point slightly before the end -> the end
        ax.annotate("",
                    xy=(x_end, y_end),
                    xytext=(x_back, y_back),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=mutation_scale))

    def draw_symmetry(color="darkturquoise"):
        ax.text(1.8, -0.5, r"$[C_2||C_{4z}t]$", ha="center", va="center", color=color)

        add_circular_arrow(center=(3, -1), radius=0.25, theta1=90, theta2=180, color=color)
        add_circular_arrow(center=(3.5, -0.5), radius=0.25, theta1=135, theta2=225, color=color)
        add_circular_arrow(center=(3.5, -1.5), radius=0.25, theta1=135, theta2=225, color=color)

    def draw_sublattice_labels():
        ax.text(0, -1.15, "SL A", ha="center", va="top", color="blue")
        ax.text(0, -2.15, "SL B", ha="center", va="top", color="red")

        ax.text(0, -1.45, r"spin up $\uparrow$", size=10, ha="center", va="top", color="blue")
        ax.text(0, -2.45, r"spin down $\downarrow$", size=10, ha="center", va="top", color="red")

    draw_symmetry()
    draw_exchange_interactions(text_size=12)
    draw_wigner_seitz()
    draw_points()
    draw_lattice_vectors()
    draw_lattice_constant(0.4)
    draw_axes(np.array([4.6, -4.6]), 1.9)
    draw_sublattice_labels()

    ax.set_xlim(-0.8, 5.8)
    ax.set_ylim(-5.673, 0.2701)

    # print(ax.get_xlim())
    # print(ax.get_ylim())

    ax.set_aspect("equal")
    ax.set_axis_off()

    if save_name:
        fig.savefig(save_name, bbox_inches='tight', pad_inches=0)

    plt.show()


# %% Spin Configuration tilted/nontilted

def generate_tilted_config(ax, kwargs_empty, kwargs_A, kwargs_B, point_kwargs, size_i, size_j):
    for i in range(size_i):
        for j in range(size_j):
            if (i + j) % 2 == 1:
                kwargs = kwargs_empty
            elif i % 2 == 0:
                kwargs = kwargs_A
            else:
                kwargs = kwargs_B

            ax.plot(i, j, **point_kwargs, **kwargs)


def generate_nontilted_config(ax, kwargs_empty, kwargs_A, kwargs_B, point_kwargs, size_i, size_j):
    for i in range(size_i):
        for j in range(size_j):
            if (i + j) % 2 == 1:
                kwargs = kwargs_A
            else:
                kwargs = kwargs_B

            ax.plot(i, j, **point_kwargs, **kwargs)


def create_imagegrid_config(tilted, kwargs_empty, kwargs_A, kwargs_B, point_kwargs):
    generator_func = generate_tilted_config if tilted else generate_nontilted_config
    kwargs_empty = kwargs_empty or dict()

    fig = plt.figure(figsize=[6.5, 6])
    grid = ImageGrid(
        fig, 111, nrows_ncols=(2, 2),
        axes_pad=(0.2, 0.2),  # (horizontal, vertical) pad in inches → equal visually
        share_all=False,
        label_mode='L'  # show labels only on left & bottom
    )
    axs = np.array([ax for ax in grid]).reshape(2, 2)

    for ax in axs.flat:
        ax.set_aspect('equal', 'box')
        generator_func(ax, kwargs_empty, kwargs_A, kwargs_B, point_kwargs, 4, 4)
        ax.set_xlim(-0.8, 3.8)
        ax.set_ylim(-0.8, 3.8)

    for ax in [axs[0, 1], axs[1, 1]]:
        ax.tick_params(left=False)
        ax.spines.left.set_visible(False)
    for ax in [axs[0, 0], axs[0, 1]]:
        ax.tick_params(bottom=False)
        ax.spines.bottom.set_visible(False)
    # for ax in axs.flat:
    #     ax.spines.top.set_visible(False)
    #     ax.spines.right.set_visible(False)
    axs[1, 0].spines.top.set_visible(False)
    axs[1, 1].spines.top.set_visible(False)
    axs[0, 0].spines.right.set_visible(False)
    axs[1, 0].spines.right.set_visible(False)
    axs[0, 0].set_yticks([0, 1, 2, 3], labels=["$N_j - 4$", "$N_j - 3$", "$N_j - 2$", "$N_j - 1$"], rotation=45)
    axs[1, 0].set_xticks([0, 1, 2, 3])
    axs[1, 1].set_xticks([0, 1, 2, 3], labels=["$N_i - 4$", "$N_i - 3$", "$N_i - 2$", "$N_i - 1$"], rotation=-45)

    d = .5  # size of break diagonal
    kwargs = dict(marker=[(-1., -d), (1., d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)

    # axis breaks
    for i in [0, 1]:
        axs[0, i].plot([i, ], [0, ], transform=axs[0, i].transAxes, **kwargs)
        axs[1, i].plot([i, ], [1, ], transform=axs[1, i].transAxes, **kwargs)
    kwargs['marker'] = [(d, 1.), (-d, -1.)]
    for i in [1, 0]:
        axs[i, 0].plot([1, ], [1 - i, ], transform=axs[i, 0].transAxes, **kwargs)
        axs[i, 1].plot([0, ], [1 - i, ], transform=axs[i, 1].transAxes, **kwargs)

    # axis labels
    x_dir = "110" if tilted else "100"
    y_dir = "-110" if tilted else "010"
    fig.text(0.5, 0.01, rf"$i \parallel \hkl[{x_dir}]$", ha="center", va="bottom", size='large')
    fig.text(0.01, 0.5, rf"$j \parallel \hkl[{y_dir}]$", ha="left", va="center", size='large', rotation=90)

    return fig, axs, grid


def average_area_bracket(ax, pos0, pos1, horizontal, kwargs=dict(color="green", marker="", linewidth=2),
                         dist=0.4, reach_solid=0, reach_dotted=0.8, reach_looselydotted=1.6):
    solid_bracket = ([reach_solid, -dist, -dist, reach_solid], [pos1 + dist, pos1 + dist, pos0 - dist, pos0 - dist])
    dotted_line1 = ([reach_solid, reach_dotted], [pos1 + dist, pos1 + dist])
    dotted_line0 = ([reach_solid, reach_dotted], [pos0 - dist, pos0 - dist])
    ldotted_line1 = ([reach_dotted, reach_looselydotted], [pos1 + dist, pos1 + dist])
    ldotted_line0 = ([reach_dotted, reach_looselydotted], [pos0 - dist, pos0 - dist])

    if not horizontal:
        solid_bracket = solid_bracket[::-1]
        dotted_line1 = dotted_line1[::-1]
        dotted_line0 = dotted_line0[::-1]
        ldotted_line1 = ldotted_line1[::-1]
        ldotted_line0 = ldotted_line0[::-1]

    ax.plot(*solid_bracket, **kwargs, linestyle="-")
    ax.plot(*dotted_line0, **kwargs, linestyle="dotted")
    ax.plot(*dotted_line1, **kwargs, linestyle="dotted")
    ax.plot(*ldotted_line1, **kwargs, linestyle=(0, (1, 5)))
    ax.plot(*ldotted_line0, **kwargs, linestyle=(0, (1, 5)))


def spin_config_tilted(save_path=None):
    kwargs_empty = dict(color="grey", alpha=0.5, markeredgecolor="k", markersize=5)
    kwargs_A = dict(color="blue", markeredgecolor="k", markersize=8)
    kwargs_B = dict(color="red", markeredgecolor="k", markersize=8)

    point_kwargs = dict(linestyle="", marker="o")

    fig, axs, grid = create_imagegrid_config(True, kwargs_empty, kwargs_A, kwargs_B, point_kwargs)
    fig.suptitle("Tilted configuration")

    # Interaction energies
    def draw_exchange_interactions(ax, color_J1="green", color_J2_1="magenta", color_J2_2="purple", text_size=12):
        shared_kwargs = dict(marker="", linestyle="-", zorder=1)
        paths_x = dict(
            J1=[[0, 2], [0, 2]],
            J2_1=[[-1, 3], [2, 2]],
            J2_2=[[-1, 2], [1, 1]]
        )
        paths_y = dict(
            J1=[[2, 0], [0, 2]],
            J2_1=[[1, 1], [2, -1]],
            J2_2=[[2, 2], [3, -1]]
        )
        paths_kwargs = dict(
            J1=[dict(color=color_J1, lw=2), dict(color=color_J1, lw=2)],
            J2_1=[dict(color=color_J2_1, lw=1.8), dict(color=color_J2_1, lw=1.8)],
            J2_2=[dict(color=color_J2_2, lw=1.8), dict(color=color_J2_2, lw=1.8)]
        )

        for J in paths_x:
            for path_x, path_y, path_kwargs in zip(paths_x[J], paths_y[J], paths_kwargs[J]):
                ax.plot(path_x, path_y, **path_kwargs, **shared_kwargs)

        # Text
        previous_font_size = mpl.rcParams["font.size"]
        mpl.rcParams["font.size"] = text_size

        box = dict(
            boxstyle="circle",
            facecolor="white",  # background color
            alpha=0.7,  # transparency (0 = transparent, 1 = opaque)
            edgecolor="none",  # no border
            pad=0.2  # small padding around text
        )
        ax.text(0.5, 0.5, "$J_1$", va="center", ha="center", color=color_J1, bbox=box)
        ax.text(2, 1, "$J_2$", va="center", ha="center", color=color_J2_1, bbox=box)
        ax.text(1, 2, "$J_2'$", va="center", ha="center", color=color_J2_2, bbox=box)

        mpl.rcParams["font.size"] = previous_font_size

    draw_exchange_interactions(axs[0, 1])

    # Unit cell
    color = "darkturquoise"
    pos_x = 0
    pos_y = 2
    axs[1, 0].plot([pos_x - 0.5, pos_x - 0.5, pos_x + 1.5, pos_x + 1.5, pos_x - 0.5],
                   [pos_y - 0.5, pos_y + 1.5, pos_y + 1.5, pos_y - 0.5, pos_y - 0.5], color=color, marker="",
                   linestyle="-", linewidth=3, solid_capstyle='round')
    axs[1, 0].plot(pos_x + 0.5, pos_y + 0.5, marker="o", markersize=4, color=color)
    fig.text(0.5, 1, "unit cell", color=color, transform=axs[1, 0].transAxes,
             ha="center", va="center", clip_on=False)

    average_area_bracket(axs[0, 0], 2, 3, True)
    axs[0, 0].text(0.5, 2.5, "profile averaging", color="green", va="center", ha="left")
    average_area_bracket(axs[1, 1], 0, 1, False)
    average_area_bracket(axs[1, 1], 2, 3, False)

    # Lattice constant
    def draw_lattice_constant(ax, distance=0.2):
        arrow_kwargs = dict(shrinkA=0, shrinkB=0, mutation_scale=3, lw=1, arrowstyle="|-|", color="k")

        bar = FancyArrowPatch((2 + distance, 0 - distance), (3 + distance, 1 - distance), **arrow_kwargs)
        ax.add_patch(bar)
        bar = FancyArrowPatch((0, 0 - distance * 1.4), (1, 0 - distance * 1.4), **arrow_kwargs)
        ax.add_patch(bar)

        ax.text(2.5 + distance - 0.06, 0.5 - distance + 0.06, r"$a$", va="bottom", ha="right")
        ax.text(0.5, 0 - distance * 1.4 - 0.1, r"$\tilde{a}$", va="top", ha="center")

    draw_lattice_constant(axs[1, 0])

    if save_path:
        fig.savefig(save_path)

    plt.show()


def spin_config_nontilted(save_path=None):
    kwargs_A = dict(color="blue", markeredgecolor="k", markersize=8)
    kwargs_B = dict(color="red", markeredgecolor="k", markersize=8)

    point_kwargs = dict(linestyle="", marker="o")

    fig, axs, grid = create_imagegrid_config(False, None, kwargs_A, kwargs_B, point_kwargs)
    fig.suptitle("Aligned configuration")

    def draw_exchange_interactions(ax, color_J1="green", color_J2_1="magenta", color_J2_2="purple", text_size=12):
        shared_kwargs = dict(marker="", linestyle="-", zorder=1)
        paths_x = dict(
            J1=[[1, 3], [2, 2]],
            J2_1=[[1, 3], [0, 2]],
            J2_2=[[1, 3], [1, 3]]
        )
        paths_y = dict(
            J1=[[2, 2], [1, 3]],
            J2_1=[[3, 1], [1, 3]],
            J2_2=[[1, 3], [2, 0]]
        )
        paths_kwargs = dict(
            J1=[dict(color=color_J1, lw=2), dict(color=color_J1, lw=2)],
            J2_1=[dict(color=color_J2_1, lw=1.8), dict(color=color_J2_1, lw=1)],
            J2_2=[dict(color=color_J2_2, lw=1.8), dict(color=color_J2_2, lw=1)]
        )

        for J in paths_x:
            for path_x, path_y, path_kwargs in zip(paths_x[J], paths_y[J], paths_kwargs[J]):
                ax.plot(path_x, path_y, **path_kwargs, **shared_kwargs)

        # Text
        previous_font_size = mpl.rcParams["font.size"]
        mpl.rcParams["font.size"] = text_size

        box = dict(
            boxstyle="circle",
            facecolor="white",  # background color
            alpha=0.7,  # transparency (0 = transparent, 1 = opaque)
            edgecolor="none",  # no border
            pad=0.2  # small padding around text
        )
        ax.text(2.5, 2, "$J_1$", va="center", ha="center", color=color_J1, bbox=box)
        ax.text(1.5, 2.5, "$J_2$", va="center", ha="center", color=color_J2_1, bbox=box)
        ax.text(1.5, 1.5, "$J_2'$", va="center", ha="center", color=color_J2_2, bbox=box)

        mpl.rcParams["font.size"] = previous_font_size

    draw_exchange_interactions(axs[0, 1])

    def draw_lattice_constant(ax, distance=0.2):
        arrow_kwargs = dict(shrinkA=0, shrinkB=0, mutation_scale=3, lw=1, arrowstyle="|-|", color="k")

        bar = FancyArrowPatch((2, 0 - distance * 1.4), (3, 0 - distance * 1.4), **arrow_kwargs)
        ax.add_patch(bar)

        ax.text(2.5, 0 - distance - 0.1, r"$a$", va="top", ha="center")

    draw_lattice_constant(axs[1, 0])

    def draw_unit_cell(ax, pos_x, pos_y, color="darkturquoise", **kwargs):
        ax.plot([pos_x - 0.5, pos_x + 0.5, pos_x + 1.5, pos_x + 0.5, pos_x - 0.5],
                [pos_y + 0.5, pos_y + 1.5, pos_y + 0.5, pos_y - 0.5, pos_y + 0.5], marker="",
                solid_capstyle='round', color=color, **kwargs)
        ax.plot(pos_x + 0.5, pos_y + 0.5, marker="o", markersize=2, color=color)

    # Unit cell
    color = "darkturquoise"
    draw_unit_cell(axs[1, 0], 0, 2, color=color, linestyle="-", linewidth=3)
    draw_unit_cell(axs[1, 0], 0, 0, color=color, linestyle="-", linewidth=2)
    draw_unit_cell(axs[1, 0], 1, 1, color=color, linestyle="-", linewidth=1)
    draw_unit_cell(axs[1, 0], 1, 3, color=color, linestyle="-", linewidth=1)
    draw_unit_cell(axs[1, 0], -1, 1, color=color, linestyle=":", linewidth=2)
    # draw_unit_cell(axs[1, 0], 1, -1, color=color, linestyle=":", linewidth=2)

    fig.text(0.5, 0.0, "unit cells", color=color, transform=axs[0, 0].transAxes,
             ha="center", va="center", clip_on=False)

    average_area_bracket(axs[0, 0], 3, 3, True)
    axs[0, 0].text(1.6, 2.5, "profile averaging", color="green", va="center", ha="left", clip_on=False)
    average_area_bracket(axs[1, 1], 2, 2, False)
    average_area_bracket(axs[1, 1], 3, 3, False)

    if save_path:
        fig.savefig(save_path)

    plt.show()


# %% Main


if __name__ == '__main__':
    # llg_equation("out/theoretical_figures/llg_equation.pdf")
    # llg_equation("out/theoretical_figures/llg_equation_T.pdf", 2)

    # dmi1("out/theoretical_figures/dmi1.pdf")

    # dmi2("out/theoretical_figures/dmi2.pdf")

    # spin_waves("out/theoretical_figures/spin_wave.pdf")

    # afm_modes("out/theoretical_figures/afm_modes.pdf")

    # toy_model("out/theoretical_figures/toy_model.pdf")

    spin_config_tilted("out/theoretical_figures/config_tilted.pdf")
    spin_config_nontilted("out/theoretical_figures/config_nontilted.pdf")

# %% cropping testing

save_name = "out/theoretical_figures/toy_model.pdf"

test = False

if test:
    crop_pdf_to_content(save_name,
                        f"{save_name[:-4]}_cropped.pdf",
                        margin_y0=250, margin_y1=0, margin_x0=100, margin_x1=150)
