import numpy as np 
from lattice import Lattice
from solver import solve

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


primitive_cell = np.array([[1.0, -0.5], [0.0, 0.5*np.sqrt(3)]])
n_kpoints = np.array([10, 10])
n_basis = np.array([100, 100])
hexagonal_lattice = Lattice(primitive_cell, n_kpoints, n_basis)
hexagonal_fourier_grids = hexagonal_lattice.fourier_grids()

ne = 2
V0 = 10
V_KxKy = test_Vq(n_basis, V0)
mu, Ek, density = solve(hexagonal_lattice, V_KxKy, ne)