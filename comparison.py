import numpy as np
from numpy.fft import fftshift, ifftshift, fft2, ifft

import OAP
import phase_test


def compare_methods(F=3, h=-0.1, nperp=(1e-3, 0), Nstep=50, plot=True):
    parabola = OAP.OAP()
    # Normalised beam width is directly related to f-number
    W = 1/(4*F)

    nx, ny = nperp

    laser = OAP.Laser(nx, ny, W)
    k = 2*np.pi / laser.wavelength

    u0, v0 = np.ogrid[-30:30:1001j, -30:30:1001j]
    E0 = laser.E_field(u0*laser.W, v0*laser.W)
    phi = k * phase_test.phase(laser.n[0], laser.n[1], u0*laser.W, v0*laser.W, h, 0)

    Cuu, Cvv, Cuv = phase_test.Cab(laser.n[0], laser.n[1], h, 0)

    print(f'C = [{Cuu}, {Cuv}]\n   [{Cuv}, {Cvv}]')

    foclen = -1 / (Cuu + Cvv)
    print(f' Focal 2 = {foclen}')

    phif = k * (Cuu + Cvv)/2 * (u0**2 + v0**2) * laser.W**2


    phiastg = k * ((Cuu - Cvv)/2 * (u0**2 - v0**2) + 2*Cuv*u0*v0) * laser.W**2
    phicoma = k * (u0**2 + v0**2) * laser.W**2 * -(u0 * nx + v0 * ny) * laser.W

    phiab = phi - phif
    phiab = phiastg + phicoma

    E_approx = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E0 * np.exp(1j*phiab))))

    du0 = np.diff(u0, axis=0).mean()
    dv0 = np.diff(v0, axis=1).mean()

    ku = np.pi/(du0*laser.W) * np.linspace(-1,1,u0.shape[0])
    kv = np.pi/(dv0*laser.W) * np.linspace(-1,1,v0.shape[1])

    u_approx = ku[:,None] / (2*np.pi) * foclen
    v_approx = kv[None,:] / (2*np.pi) * foclen

    (xf, yf, zf), nf = phase_test.focal_position(laser.n[0], laser.n[1], h, 0)
    U, V, N = OAP.get_basis(-nf[0], -nf[1], -nf[2])

    u, v = np.ogrid[-50:50:201j, -50:50:201j]

    x0 = xf + (u * U[0] + v * V[0]) * laser.wavelength
    y0 = yf + (u * U[1] + v * V[1]) * laser.wavelength
    z0 = zf + (u * U[2] + v * V[2]) * laser.wavelength

    Efoc = parabola.propagate(laser, x0, y0, z0, h, Nstep, Nstep, True)

    if plot:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(np.rot90(abs(Efoc)**2), extent=(u[0,0], u[-1,-1], v[0,0], v[-1,-1]))
        axs[1].imshow(np.rot90(abs(E_approx)**2), extent=(u_approx[0,0], u_approx[-1,-1], v_approx[0,0], v_approx[-1,-1]))
        axs[0].set_xlim(max(u[0,0], u_approx[0,0]), min(u[-1,-1], u_approx[-1,-1]))
        axs[1].set_xlim(max(u[0,0], u_approx[0,0]), min(u[-1,-1], u_approx[-1,-1]))
        axs[0].set_ylim(max(v[0,0], v_approx[0,0]), min(v[-1,-1], v_approx[-1,-1]))
        axs[1].set_ylim(max(v[0,0], v_approx[0,0]), min(v[-1,-1], v_approx[-1,-1]))


    return (u, v, Efoc), (u_approx, v_approx, E_approx)


def compare_ud(F=3, h=-0.1, nperp=(1e-3, 0), Nstep=50):
    parabola = OAP.OAP()
    # Normalised beam width is directly related to f-number
    W = 1/(4*F)

    nx, ny = nperp

    laser = OAP.Laser(nx, ny, W)
    k = 2*np.pi / laser.wavelength

    u0, v0 = np.ogrid[-30:30:1001j, -30:30:1001j]
    E0 = laser.E_field(u0*laser.W, v0*laser.W)
    phi = k * phase_test.phase(laser.n[0], laser.n[1], u0*laser.W, v0*laser.W, h, 0)

    Cuu, Cvv, Cuv = phase_test.Cab(laser.n[0], laser.n[1], h, 0)

    print(f'C = [{Cuu}, {Cuv}]\n   [{Cuv}, {Cvv}]')

    foclen = -1 / (2 * (Cuu + Cvv))

    phif = k * (Cuu + Cvv)/2 * (u0**2 + v0**2) * laser.W**2


    phiastg = k * ((Cuu - Cvv)/2 * (u0**2 - v0**2) + 2*Cuv*u0*v0) * laser.W**2
    #phicoma = k * (u0**2 + v0**2) * laser.W**2 * -(u0 * nx + v0 * ny) * laser.W

    phiab = phi - phif
    # phiab = phiastg + phicoma

    E_approx = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E0 * np.exp(1j*phiab))))

    du0 = np.diff(u0, axis=0).mean()
    dv0 = np.diff(v0, axis=1).mean()

    ku = np.pi/(du0*laser.W) * np.linspace(-1,1,u0.shape[0])
    kv = np.pi/(dv0*laser.W) * np.linspace(-1,1,v0.shape[1])

    u_approx = ku / (2*np.pi) * foclen
    v_approx = kv / (2*np.pi) * foclen

    (xf, yf, zf), nf = phase_test.focal_position(laser.n[0], laser.n[1], h, 0)
    U, V, N = OAP.get_basis(*nf)

    d, u = np.ogrid[-100:100:101j, -50:50:101j]

    x0 = xf + (u * U[0] + d * N[0]) * laser.wavelength
    y0 = yf + (u * U[1] + d * N[1]) * laser.wavelength
    z0 = zf + (u * U[2] + d * N[2]) * laser.wavelength 
    
    Efoc = parabola.propagate(laser, x0, y0, z0, h, Nstep, Nstep, True)

    return (u, d, Efoc), (u_approx, v_approx, E_approx)


def propagate_asm(u, v, E, z):
    du = np.diff(u, axis=0).mean()
    dv = np.diff(v, axis=1).mean()

    ku_on_k0, kv_on_k0, dz = np.ogrid[-1/(2*du):1/(2*du):u.shape[0]*1j, -1/(2*dv):1/(2*dv):v.shape[1]*1j, z.min():z.max():z.size*1j]

    print(ku_on_k0.shape, kv_on_k0.shape, dz.shape)
    Eff = fftshift(fft2(ifftshift(E)))
    Eff_slice = (Eff[:,:,None] * np.exp(2j * np.pi * (np.sqrt(1+0j - ku_on_k0**2 - kv_on_k0**2).real - 1) * dz)).sum(axis=1)
    print(Eff_slice.shape)
    Eprop = fftshift(ifft(ifftshift(Eff_slice), axis=0))

    return Eprop


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    A, B = compare_methods(F=3, nperp=(0.707e-3,0.707e-3), h=0.2, Nstep=50, plot=False)
    dz = np.linspace(-200,200,101)
    Aprop = propagate_asm(*A, dz)
    Bprop = propagate_asm(*B, dz)

    fig, axs = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 4]}, figsize=(8,4), constrained_layout=True)
    axs[0,0].imshow((abs(A[2])**2), extent=(A[0].min(), A[0].max(), A[1].min(), A[1].max()))
    axs[1,0].imshow((abs(B[2])**2), extent=(B[0].min(), B[0].max(), B[1].min(), B[1].max()))
    axs[1,0].set_xlim(axs[0,0].get_xlim())
    axs[1,0].set_ylim(axs[0,0].get_ylim())

    axs[0,1].imshow(abs(Aprop)**2, extent=(-200,200,A[0].min(),A[0].max()))
    axs[1,1].imshow(abs(Bprop)**2, extent=(-200,200,B[0].min(),B[0].max()))
    axs[1,1].set_ylim(axs[0,1].get_ylim())
    axs[0,1].plot([0,0], axs[0,1].get_ylim(), 'w--')
    axs[1,1].plot([0,0], axs[1,1].get_ylim(), 'w--')

    for ax in axs[:,0]:
        ax.set_xlabel('v/$\lambda$')
    for ax in axs[:,1]:
        ax.set_xlabel('$\delta$/$\lambda$')
    for ax in axs.flat:
        ax.set_ylabel('u/$\lambda$')
        ax.label_outer()

    plt.savefig('focusing.svg')
    plt.show()
