import numpy as np

class Lattice(object):
    """
    2D lattice class
    """
    def __init__(self, primitive_cell, n_kpoints, n_basis):
        """
        :primitive cell: np.ndarray, each column represents a direction
        """
        self.primitive_cell = primitive_cell
        self._reciprocal()
        self.n_kpoints = n_kpoints
        self.n_basis = n_basis

    def _reciprocal(self):
        Rotation = np.array([[0.0, -1.0], [1.0, 0.0]])   # rotate a vector 90 degree anti-clock wise
        a1 = self.primitive_cell[:, 0]
        a2 = self.primitive_cell[:, 1]
        b1 = (2*np.pi) * (Rotation @ a2) / np.dot(a1, (Rotation @ a2))
        b2 = (2*np.pi) * (Rotation @ a1) / np.dot(a2, (Rotation @ a1))
        self.reciprocal_cell = np.vstack([b1, b2]).T

    def brillouin_zone(self):
        """
        :n_kpoints: # of k points in each direction
        """
        bs = self.reciprocal_cell
        Ix_Bz = np.linspace(0, 1, self.n_kpoints[0])
        Iy_Bz = np.linspace(0, 1, self.n_kpoints[1])
        Ixx_Bz, Iyy_Bz = np.meshgrid(Ix_Bz, Iy_Bz)
        Ixy_Bz = np.vstack([Ixx_Bz.ravel(), Iyy_Bz.ravel()])
        return bs @ Ixy_Bz

    def fourier_grids(self):
        """
        :n_basis: # of basis per direction
        """
        bs = self.reciprocal_cell
        Ix_fourier = np.arange(-self.n_basis[0]//2, self.n_basis[0]//2, 1)
        Iy_fourier = np.arange(-self.n_basis[1]//2, self.n_basis[1]//2, 1)
        Ixx_fourier, Iyy_fourier = np.meshgrid(Ix_fourier, Iy_fourier)
        Ixy_fourier = np.vstack([Ixx_fourier.ravel(order='F'), Iyy_fourier.ravel(order='F')])   # unroll in a column major order
        return bs @ Ixy_fourier

    def realSpace_zone(self, n_points):
        """
        :n_points: # of real space samples per direction
        """
        a0s = self.primitive_cell
        Ix = np.linspace(0, 1, n_points[0])
        Iy = np.linspace(0, 1, n_points[1])
        Ixx, Iyy = np.meshgrid(Ix, Iy)
        Ixy = np.vstack([Ixx.ravel(), Iyy.ravel()])
        return a0s @ Ixy

    def high_symmetry_path(self, high_sym_points):
        """
        :high_sym_points: 2 * n_points (ordered)
        """
        bs = self.reciprocal_cell
        num_sym_points = high_sym_points.shape[1]
        path = []
        for i in range(num_sym_points-1):
            start = high_sym_points[:, i]
            stop = high_sym_points[:, i+1]
            line = np.vstack([np.linspace(start[0], stop[0], 20), np.linspace(start[1], stop[1], 20)])
            path.append(line)
        path = np.hstack(path)
        return bs @ path
