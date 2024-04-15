import numpy as np
from math import factorial
import matplotlib.pyplot as plt
from scipy.special import sph_harm
import scipy.special as sp

N = 50

def radial_function(n, l, r, a0):
    """ Compute the normalized radial part of the wavefunction using
    Laguerre polynomials and an exponential decay factor.

    Args:
        n (int): principal quantum number
        l (int): azimuthal quantum number
        r (numpy.ndarray): radial coordinate
        a0 (float): scaled Bohr radius
    Returns:
        numpy.ndarray: wavefunction radial component
    """

    laguerre = sp.genlaguerre(n - l - 1, 2 * l + 1)
    p = 2 * r / (n * a0)

    constant_factor = np.sqrt(
        ((2 / n * a0) ** 3 * (sp.factorial(n - l - 1))) /
        (2 * n * (sp.factorial(n + l)))
    )
    return constant_factor * np.exp(-p / 2) * (p ** l) * laguerre(p)


def Psi_sol(n,l,m):
    a0 = 1e-1

    return lambda r_, theta_, phi_: radial_function(
        n, l, r_, a0
    ) * angular_function(
        m, l, theta_, phi_
    )

def angular_function(m, l, theta, phi):
    """ Compute the normalized angular part of the wavefunction using
    Legendre polynomials and a phase-shifting exponential factor.

    Args:
        m (int): magnetic quantum number
        l (int): azimuthal quantum number
        theta (numpy.ndarray): polar angle
        phi (int): azimuthal angle
    Returns:
        numpy.ndarray: wavefunction angular component
    """

    legendre = sp.lpmv(m, l, np.cos(theta))

    constant_factor = ((-1) ** m) * np.sqrt(
        ((2 * l + 1) * sp.factorial(l - np.abs(m))) /
        (4 * np.pi * sp.factorial(l + np.abs(m)))
    )
    return constant_factor * legendre * np.real(np.exp(1.j * m * phi))


def sphere_to_cartesian(r, theta, phi):
    return r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)


r = np.linspace(0, 2, N)
theta = np.linspace(0, np.pi, N)
phi = np.linspace(0, 2 * np.pi, 2*N)
Rad, Theta, Phi = np.meshgrid(r, theta, phi)
# Psi = lambda

fig = plt.figure(figsize=(20,20))
plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Serif"],
        }
    )
# axs = fig.add_subplot(111,projection='3d')
X, Y, Z = sphere_to_cartesian(Rad, Theta, Phi)
#
# Psi_nlm = Psi_sol(2,0,0)
# pdf = np.abs(Psi_nlm(Rad, Theta, Phi)) ** 2
# pdf_clip = pdf.clip(0,pdf.max()/40)
# axs.scatter(X, Y, Z, c=pdf_clip, s=2, alpha = 0.05, marker='o', cmap="PuRd")
# axs.grid(False)
# axs.axis("off")
# plt.tight_layout()
# plt.show()
sub = 1
for n in range(2,4):
    for l in range(n):
        for m in range(l+1):
            print(n,l,m)

            axs = fig.add_subplot(3,3,sub,projection='3d')
            axs.set_title(f"{n}, {l}, {m}", fontsize=50)
            sub += 1

            Psi_nlm = Psi_sol(n,l,m)
            pdf = np.abs(Psi_nlm(Rad, Theta, Phi)) ** 2
            pdf_clip = pdf.clip(0, pdf.max() / 40)
            axs.scatter(X, Y, Z, c=pdf_clip, s=2, alpha=0.05, marker='o', cmap="PuRd")

            axs.grid(False)
            axs.axis("off")
            plt.tight_layout()
# plt.show()
plt.savefig("../orbit_plots/hydrogen_orbit.png", dpi=600)