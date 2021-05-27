import numpy as np

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
    
    for i, p in enumerate(psi):
        A_p = V.T @ (c * np.exp(-1j * s * (p - psi0))) @ U
        print(i)
        yield A_p

def propagate_slices(x, y, A, psi, slices, psi0=-np.pi/2, bandlimit=None):
    A_all_sliced = [ np.zeros(psi.shape + A[slc].shape, dtype='complex') for slc in slices ]
    for n, A in enumerate(propagate_iter(x,y,A,psi,psi0,bandlimit)):
        for A_sliced, slc in zip(A_all_sliced, slices):
            A_sliced[n] = A[slc]

    return A_all_sliced

def propagate_to(x, y, A, psi, psi0=-np.pi/2, bandlimit=None):
    # x: (n_x,) 1d x axis
    # y: (n_y,) 1d y axis
    # A: (n_x, n_y) 2d complex field amplitude. Note 'ij' meshgrid indexing.
    # psi: (n_psi,) points to evaluate A(psi) at

    return tuple(propagate_iter(x,y,A,psi,psi0,bandlimit))

def main():
    import plotting
    import matplotlib.pyplot as plt
    x = np.linspace(-5,5,1001)
    y = np.linspace(-5,5,1001)
    X, Y = np.meshgrid(x, y, indexing='ij')
    R2 = X**2 + Y**2
    A = (R2 < 2**2) * (R2 > 1**2)
    # For example, x, y, W and f here are in cm
    # and wavelength is in microns
    # so axes on the plots are in microns
    plotting.plot_focus(x, y, A, 1, 100, 1, 1000)
    plt.show()

if __name__ == '__main__':
    main()
