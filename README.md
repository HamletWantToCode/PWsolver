# PWsolve
Schrodinger equation solver using plane wave method, currently support 2D system, will be extended to include 1D and 3D cases.

## To do
- [ ] write test cases and enable pytest
- [ ] support 1D system

## Changelog
### 0.0.2 - 2019.09.17
- solver.py:
    - `hamilton`: replace full numpy array with `scipy.sparse` matrix
    - `solve`: replace the `np.linalg.eigh` by `scipy.sparse.linalg.eigsh`
### 0.0.1 - 2019.08.31
- lattice.py: create 2D Bravais lattice with given primitive cell, parameters include `primitive_cell`, `n_kpoints` (used to build up Brillouine zone), `n_basis` (used to build up reciprocal space grids for FFT).
- solver.py: 
    - `hamilton`: function to build Hamiltonian matrix, acceptes `V_KxKy` (potential in reciprocal space), `k` (Brillouine zone vector), `fourier_grids` (reciprocal space grids).
    - `solve`: diagonalize the Hamiltonian matrix to compute electron density in reciprocal space and kinetic energy (per cell).