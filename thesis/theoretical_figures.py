import numpy as np
import thesis.mpl_configuration

import matplotlib.pyplot as plt

# draw a vector
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

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
        return np.min(zs)  # Used for depth sorting


# %%
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

    thermal_noise_amplitude = thermal_noise      # Change this value to get stochastic

    def a(t):
        return dA0 * np.exp(-t / tau) + A
    def Sx(t):
        return a(t) * np.cos(w*t)
    def Sy(t):
        return a(t) * np.sin(w*t)
    def Sz(t):
        return np.sqrt(1 - a(t)**2)
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
            S[i] = S[i-1] + dS(S[i-1], H(t_arr[i-1], i), t_arr[i]-t_arr[i-1])
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

    u, v = np.mgrid[0:2 * np.pi:20j, 0:(np.pi/2):10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="grey", alpha=0.5, linewidth=0.1)

    # ax.plot(Sx(t), Sy(t), Sz(t), linestyle="--")

    S_L = S_llg(S(0), t)
    # S_L = S_llg(np.array([0.0, 0.0, 1.0]), t)
    ax.plot(S_L[:,0], S_L[:,1], S_L[:,2], linestyle="--", linewidth=0.8, color="#00a9e0")

    t0 = -0.2
    i_t = np.argmin(np.abs(t - t0))

    H0_arrow = Arrow3D([- 0.0 * H0[0], 1.2 * H0[0]], [- 0.0 * H0[1], 1.2 * H0[1]], [- 0.0 * H0[2], 1.2 * H0[2]],
                       mutation_scale=20, shrinkA=0, shrinkB=0, lw=1, arrowstyle="-|>", color="k",)
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
    S_arrow = Arrow3D([0, S_L[i_t,0]], [0, S_L[i_t,1]], [0, S_L[i_t,2]], mutation_scale=20,
                      shrinkA=0, shrinkB=0, lw=1, arrowstyle="-|>", color="k",)
    ax.add_artist(S_arrow)
    ax.text(*(S_L[i_t] - np.array([0.02, 0.02, 0.1])), r'$\vec{S}_i$', size=font_size, ha="left", va="top")

    # damping_vec = damping(S(t0), H(t0))
    damping_vec = damping(S_L[i_t], H(t0, i_t))
    damping_vec *= 0.3 / np.sqrt(damping_vec.dot(damping_vec))
    # damping_vec += S(t0)
    damping_vec += S_L[i_t]
    # damping_arrow = Arrow3D([Sx(t0), damping_vec[0]], [Sy(t0), damping_vec[1]], [Sz(t0), damping_vec[2]],
    #                         mutation_scale=20, shrinkA=0, shrinkB=0, lw=1, arrowstyle="-|>", color="k")
    damping_arrow = Arrow3D([S_L[i_t,0], damping_vec[0]], [S_L[i_t,1], damping_vec[1]], [S_L[i_t,2], damping_vec[2]],
                            mutation_scale=20, shrinkA=0, shrinkB=0, lw=1, arrowstyle="-|>", color="#53c412")
    ax.add_artist(damping_arrow)
    ax.text(*(damping_vec + np.array([0.07, 0.02, -0.09])), r'$- \vec{S}_i \times \left(\vec{S}_i \times \vec{H}_i \right)$',
            size=font_size, ha="right", va="top", color="#53c412")  #backgroundcolor=(1.0, 1.0, 1.0, 0.6)

    # gilbert_vec = gilbert_torque(S(t0), H(t0))
    gilbert_vec = gilbert_torque(S_L[i_t], H(t0, i_t))
    gilbert_vec *= 0.3 / np.sqrt(gilbert_vec.dot(gilbert_vec))
    # gilbert_vec += S(t0)
    gilbert_vec += S_L[i_t]
    # gilbert_arrow = Arrow3D([Sx(t0), gilbert_vec[0]], [Sy(t0), gilbert_vec[1]], [Sz(t0), gilbert_vec[2]],
    #                         mutation_scale=20, shrinkA=0, shrinkB=0, lw=1, arrowstyle="-|>", color="k")
    gilbert_arrow = Arrow3D([S_L[i_t,0], gilbert_vec[0]], [S_L[i_t,1], gilbert_vec[1]], [S_L[i_t,2], gilbert_vec[2]],
                            mutation_scale=20, shrinkA=0, shrinkB=0, lw=1, arrowstyle="-|>", color="#f47c20")
    ax.add_artist(gilbert_arrow)
    ax.text(*(gilbert_vec - np.array([0.05, 0.05, 0.08])), r'$- \vec{S}_i \times \vec{H}_i$',
            size=font_size, ha="left", va="top", color="#f47c20")

    ax.view_init(elev=20., azim=-15, roll=0)

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(0.0, 1.0)
    ax.set_box_aspect((1.0,1.0,1.0), zoom=4)
    ax.set_aspect("equal")

    ax.set_axis_off()

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)  # No border padding

    if save_name:
        fig.savefig(save_name, bbox_inches='tight')

    plt.show()

if __name__ == '__main__':
    llg_equation("out/theoretical_figures/llg_equation.pdf")
