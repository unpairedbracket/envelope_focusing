import numpy as np

class OAP:

    def __init__(self):
        pass

    def z(self, x, y):
        return (x**2 + y**2 - 1)/2

    def propagate(self, laser, x0, y0, z0, h, Nx, Ny, sinc_correction=False):
        Ux, Uy, Uz = laser.U
        Vx, Vy, Vz = laser.V
        nx, ny, nz = laser.n

        Xh = h
        Yh = 0
        Zh = self.z(Xh, Yh)

        Dr = 3 * laser.W
        X, Y = np.ogrid[Xh-Dr:Xh+Dr:Nx*1j, Yh-Dr:Yh+Dr:Ny*1j]
        dx = 2*Dr / (Nx-1)
        dy = 2*Dr / (Ny-1)

        Z = self.z(X, Y)

        u = (X-Xh)*Ux + (Y-Yh)*Uy + (Z-Zh)*Uz
        v = (X-Xh)*Vx + (Y-Yh)*Vy + (Z-Zh)*Vz
        d = (X-Xh)*nx + (Y-Yh)*ny + (Z-Zh)*nz

        E = laser.E_field(u, v)
        
        k = 2*np.pi / laser.wavelength

        E0 = np.zeros((x0+y0+z0).shape, 'complex')

        factor = 1
        for i in range(Nx):
            Xi = X[i,0]
            for j in range(Ny):
                Yj = Y[0,j]
                Zij = Z[i,j]

                r = np.sqrt((Xi - x0)**2 + (Yj - y0)**2 + (Zij - z0)**2)
                gradG_dot_nm = 2 * (1j * k * r - 1) * np.exp(1j * k * (r + d[i,j])) / r**3 * (Xi*(x0 - Xi) + Yj*(y0 - Yj) + (Zij - z0))
                dldx = nx + nz * Xi + ((Xi - x0) + Xi * (Zij - z0)) / r
                dldy = ny + nz * Yj + ((Yj - y0) + Yj * (Zij - z0)) / r
                if sinc_correction:
                    factor = np.sinc(dldx * k * dx / (2*np.pi)) * np.sinc(dldy * k * dy / (2*np.pi))
                E0 += factor * E[i,j] * gradG_dot_nm * dx * dy

        return E0 / (4 * np.pi)



class Laser:
    # Off-axis parameter
    wavelength = 1e-6

    def __init__(self, n_x, n_y, W):
        self.U, self.V, self.n = get_basis(n_x, n_y)

        self.W = W


    def E_field(self, u, v):
        return np.exp(-(u**2+v**2)/self.W**2)


def get_basis(n_x, n_y, n_z=None):
    if n_z is None:
        n_z = - np.sqrt(1 - (n_x**2+n_y**2))
    
    U = np.array([n_z, 0, -n_x]) / np.sqrt(1 - n_y**2)
    V = np.array([- n_y*n_x, n_z**2 + n_x**2, -n_y * n_z]) / np.sqrt(1-n_y**2)
    N = np.array([n_x, n_y, n_z])

    return U, V, N

