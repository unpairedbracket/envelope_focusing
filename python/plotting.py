import numpy as np
import matplotlib.pyplot as plt

import propagate


def plot_focus(x, y, A, W, focal_distance, wavelength, Nstep):
    '''
    x: vector, shape (n_x, 1) in your units of choice (e.g. cm)
    y: vector, shape (1, n_y) in your units of choice
    A: matrix, shape (n_x, n_y) the complex e-field you want to focus
    W: scalar, the `radius' of A, in the same units as x and y
    focal_distance: the focal length of the optic, same units as w
    lambda: wavelength of the laser.
    Nstep: Number of steps to take in the Guoy phase angle

    If lambda is given in different units to the macroscopic distances
    x,y,W,focal_distance then the results will be plotted in the units of 
    lambda
    '''

    guoy = np.linspace(-np.pi/2, np.pi/2, Nstep+1)
    x0 = x/W
    y0 = y/W
    A_focus = propagate.propagate_to(x0, y0, A, (0,))[0]
    # This yields the x=0 slice (or as close to it as we can get)
    i_xmin = np.argmin(abs(x0))
    A_slice = propagate.propagate_slices(x0, y0, A, guoy, slices=[np.s_[i_xmin, :]])[0]

    w0 = wavelength * focal_distance / W / np.pi
    zR = np.pi * w0**2 / wavelength
    z0 = np.tan(guoy)
    w_of_z = w0 / np.cos(guoy)

    I_focus = abs(A_focus)**2 * (W/w0)**2
    I_slice = abs(A_slice)**2 * (W/w_of_z[:,None])**2

    fig, axs = plt.subplots(1, 2)

    axs[0].imshow(I_focus.T, extent=(x0.min() * w0, x0.max() * w0, y0.min() * w0, y0.max() * w0))

    Z0, Y0 = np.meshgrid(z0, x0, indexing='ij')
    axs[1].pcolormesh(Z0 * zR, Y0 * w_of_z[:,None], I_slice, shading='gouraud')
    axs[1].set_xlim([-10*zR, 10*zR])
    axs[1].set_ylim([-10*w0, 10*w0])
