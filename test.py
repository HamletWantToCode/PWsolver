import numpy as np 
from scipy.sparse.linalg import eigsh
from lattice import Lattice
from solver import hamilton, solve
import matplotlib.pyplot as plt 


def test_Vq(n_basis, V0):
    """
    V(x) = cos(K1*x)**2 + cos(K2*x)**2 + cos((K1+K2)*x)**2
    """
    Vq = np.zeros([n_basis[1], n_basis[0]], dtype=np.complex128)  # in meshgrid we use 'xy' convention, which is in reverse order to 'ij'
    Vq[0, 1] = Vq[0, -1] = -V0 * 0.25
    Vq[1, 0] = Vq[-1, 0] = -V0 * 0.25
    Vq[1, 1] = Vq[-1, -1] = -V0 * 0.25
    Vq[0, 0] = -V0 * 1.5
    return Vq

# hexagonal cell a1 = a2 = a0
primitive_cell = np.array([[1.0, -0.5], [0.0, 0.5*np.sqrt(3)]])
n_kpoints = np.array([10, 10])
n_basis = np.array([10, 10])
hexagonal_lattice = Lattice(primitive_cell, n_kpoints, n_basis)

hexagonal_fourier_grids = hexagonal_lattice.fourier_grids()

# K point (0.5, 0.0)
V0s = np.arange(0, 2, 0.05)
k_K = (hexagonal_lattice.reciprocal_cell @ np.array([[0.5], [0.0]])).squeeze()
band_gaps = []
for V0 in V0s:
    V_KxKy = test_Vq(n_basis, V0)
    _, H_kK1K2 = hamilton(k_K, V_KxKy, hexagonal_fourier_grids)
    E_k, _ = eigsh(H_kK1K2, k=2, which='SA')
    E_LOMO, E_HOMO = np.amax(E_k), np.amin(E_k)
    band_gaps.append(E_LOMO - E_HOMO)

Nes = np.arange(1, 10, 1)
Eks = []
TF_Eks = []
zero_V_KxKy = test_Vq(n_basis, 0)
for ne in Nes:
    _, Ek, _ = solve(hexagonal_lattice, zero_V_KxKy, ne)
    Eks.append(Ek)
    TF_Eks.append(np.pi * (ne**2) * 2 / np.sqrt(3))


fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(TF_Eks, Eks, 'bo', label='Schrodinger equation')
ax1.plot(TF_Eks, TF_Eks, 'r', label='Thomas Fermi')
ax1.legend(prop={'size': 15})
ax1.set_xlabel(r'$E_k$ (Schrodinger)', fontsize=15)
ax1.set_ylabel(r'$E_k$ (TF)', fontsize=15)
ax1.set_title('Kinetic energy', fontsize=15)

ax2.plot(V0s * 0.25, 0.5 * V0s, 'r', linewidth=2, label=r'$2|\hat{V}(\mathbf{K}_1)|$')
ax2.plot(V0s * 0.25, band_gaps, 'b--', linewidth=2, label=r'$\Delta E$')
ax2.set_xlabel(r'$|\hat{V}(\mathbf{K}_1)|$', fontsize=15)
ax2.set_ylabel(r'$\Delta E$', fontsize=15)
ax2.set_title('Near free electron approximation', fontsize=15)
ax2.legend(prop={'size': 15})

plt.show()

