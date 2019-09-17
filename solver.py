import numpy as np 
from scipy.sparse import eye, kron, diags
from scipy.sparse.linalg import eigsh


def hamilton(k, V_KxKy, fourier_grids):
    """
    :k: k in Bz
    :V_KxKy: FFT of V(x) shape (n_basis[1], n_basis[0]) (unroll in column major order 'F')
    :IK_grids: (2, n_basis[1]*n_basis[0]), index of reciprocal grids
    :fourier_grids: shape (2, n_basis[1]*n_basis[0])
    """
    hamilton_size = fourier_grids.shape[1]
    Ek_kK = 0.5 * np.sum((k[:, np.newaxis] + fourier_grids)**2, axis=0)  # 100
    Hamiltonian_matrix = diags(Ek_kK) 

    n_Ky, n_Kx = V_KxKy.shape    # since we use 'xy' mode in meshgrid
    for d_Kx in range(n_Kx//2+1):
        V_Ky = V_KxKy[:, d_Kx]
        VKxi_matrix = eye(n_Ky)*V_Ky[0]
        for d_Ky in range(1, n_Ky//2+1):
            VKxi_matrix += eye(n_Ky, k=-d_Ky)*V_Ky[d_Ky].conj()
            VKxi_matrix += eye(n_Ky, k=d_Ky)*V_Ky[d_Ky]
        lower_eye_matrix = eye(n_Kx, k=-d_Kx)
        Hamiltonian_matrix += kron(lower_eye_matrix, VKxi_matrix)
        if d_Kx != 0:
            upper_eye_matrix = eye(n_Kx, k=d_Kx)
            Hamiltonian_matrix += kron(upper_eye_matrix, VKxi_matrix.H)
    return Ek_kK, Hamiltonian_matrix


def solve(lattice, V_KxKy, ne):
    brillouin_zone = lattice.brillouin_zone()
    fourier_grids = lattice.fourier_grids()
    num_cell = brillouin_zone.shape[1]
    Ek = 0
    density_matrix_size = fourier_grids.shape[1]
    density_matrix = np.zeros([density_matrix_size, density_matrix_size], dtype=np.complex128)
    
    band_energy_bottom = np.inf
    band_energy_top = -np.inf
    for k in brillouin_zone.T:
        Ek_kK, H_kK1K2 = hamilton(k, V_KxKy, fourier_grids)
        band_k, c_kKn = eigsh(H_kK1K2, k=ne, which='SA')

        band_energy_bottom = np.minimum(np.amin(band_k), band_energy_bottom)
        band_energy_top = np.maximum(np.amax(band_k), band_energy_top)

        rho_kK1K2 = c_kKn.conj() @ c_kKn.T
        density_matrix += rho_kK1K2
        Ek += np.trace(np.diag(Ek_kK) @ rho_kK1K2)
    
    mu = band_energy_top - band_energy_bottom
    n_Ky, n_Kx = V_KxKy.shape          # since we use 'xy' mode in meshgrid
    rho_rKxKy = np.zeros([n_Ky, n_Kx//2+1], dtype=np.complex128)
    for i in range(n_Kx//2+1):
        if i == 0:
            rho_rKxKy[0, 0] = np.trace(density_matrix)
        else:
            rho_rKxKy[0, i] = np.trace(density_matrix[i*n_Ky:, :-i*n_Ky])
        for j in range(1, n_Ky//2+1):
            if i == 0:
                rho_rKxKy[j, 0] = np.trace(density_matrix, -j)
                rho_rKxKy[-j, 0] = rho_rKxKy[j, 0].conj()
            else:
                rho_rKxKy[j, i] = np.trace(density_matrix[i*n_Ky:, :-i*n_Ky], -j)
    return mu, Ek.real/num_cell, rho_rKxKy/num_cell
