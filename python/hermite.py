import numpy as np

def generate_hermites(x,bandlimit=None):
    # This generates Hermite functions ψn(x / sqrt(2))
    # Which are the eigenfunctions of the focusing envelope equation

    n_func = bandlimit if bandlimit else x.size

    V = np.zeros((n_func, x.size))

    V[0,:] = np.exp(-x**2)
    V[1,:] = np.exp(-x**2) * 2 * x

    for j in np.arange(1, n_func-1):
        V[j+1, :] = np.sqrt(1 / (j+1)) * ( 2 * x * V[j, :] - np.sqrt(j) * V[j-1] )

    return V

