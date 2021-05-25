import numpy as np
import matplotlib as plt

from hermite import generate_hermites

def propagate_iter(x, y, A, psi, psi0=-np.pi/2, bandlimit=None):
    # x: (n_x,) 1d x axis
    # y: (n_y,) 1d y axis
    # A: (n_x, n_y) 2d complex field amplitude. Note 'ij' meshgrid indexing.
    # psi: (n_psi,) points to evaluate A(psi) at

    U = generate_hermites(x,bandlimit)
    V = generate_hermites(y,bandlimit)

    c = V @ A @ U.T

    s = np.arange(U.shape[0])[:, None] + np.arange(V.shape[0])[None, :]
    
    for p in psi:
        A_p = V.T @ (c * np.exp(-1j * s * (p - psi0))) @ U
        yield A_p

def propagate_slices(x, y, A, psi, slices, psi0=-np.pi/2, bandlimit=None):
    A_all_sliced = [
        [ 
            A[slc] for slc in slices 
        ] for A in propagate_iter(x,y,A,psi,psi0,bandlimit)
    ]

    return [ np.array(A_slc) for A_slc in zip(*A_all_sliced) ]

def propagate_to(x, y, A, psi, psi0=-np.pi/2, bandlimit=None):
    # x: (n_x,) 1d x axis
    # y: (n_y,) 1d y axis
    # A: (n_x, n_y) 2d complex field amplitude. Note 'ij' meshgrid indexing.
    # psi: (n_psi,) points to evaluate A(psi) at

    return tuple(propagate_iter(x,y,A,psi,psi0,bandlimit))
