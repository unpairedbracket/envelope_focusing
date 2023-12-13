import numpy as np

from numpy import sqrt

def phase(nx, ny, u, v, x_h, y_h):

    z_h = (x_h**2 + y_h**2 - 1) / 2

    nz = - sqrt(1 - nx**2 - ny**2)
    factor = sqrt(1 - ny**2)
    
    Ux = nz / factor
    Uy = 0
    Uz = -nx / factor

    Vx = ny * Uz
    Vy = nz * Ux - nx * Uz
    Vz = - ny * Ux

    cosPhi = 1/sqrt(1 + x_h**2 + y_h**2)

    nx_m = -x_h * cosPhi
    ny_m = -y_h * cosPhi
    nz_m = cosPhi

    cosPhibm = nx * nx_m + ny * ny_m + nz * nz_m

    # x, y before raytrace
    x = x_h + Ux * u + Vx * v
    y = y_h + Uy * u + Vy * v
    z = z_h + Uz * u + Vz * v

    # A d^2 - 2 B d + C = 0
    A = nx**2 + ny**2
    B = nz - x * nx + y * ny
    C = (x**2 + y**2 - 1) - 2 * z

    delta = (B + sqrt(B**2 - A * C)) / A

    delta[A < 1e-16] = (C/(2*B)*(1 + A*C/(4*B**2)))[A < 1e-16]

    dx = Ux * u + Vx * v + nx * delta
    dy = Uy * u + Vy * v + ny * delta

    phase_on_k = cosPhi * cosPhibm * (dx**2 + dy**2)

    return phase_on_k

def Cab(nx, ny, x_h, y_h):

    nz = - sqrt(1 - nx**2 - ny**2)
    factor = sqrt(1 - ny**2)
    
    Ux = nz / factor
    Uy = 0
    Uz = -nx / factor

    Vx = ny * Uz
    Vy = nz * Ux - nx * Uz
    Vz = - ny * Ux

    cosPhi = 1/sqrt(1 + x_h**2 + y_h**2)

    r_h2 = x_h**2 + y_h**2

    nx_m = -x_h * cosPhi
    ny_m = -y_h * cosPhi
    nz_m = cosPhi

    cosPhibm = nx * nx_m + ny * ny_m + nz * nz_m


    na2 = (x_h * nx - y_h * ny - nz)**2
    Cuu = cosPhi * (Vx**2 + Vy**2 + (2 * x_h * Vx + 2 * y_h * Vy + r_h2 * Vz) * Vz) / na2
    Cvv = cosPhi * (Ux**2 + Uy**2 + (2 * x_h * Ux + 2 * y_h * Uy + r_h2 * Uz) * Uz) / na2
    Cuv = -cosPhi * (Ux*Vx + Uy*Vy + x_h * (Ux*Vz+Vx*Uz) + y_h * (Uy*Vz+Vy*Uz) + r_h2 * Uz*Vz) / na2

    return cosPhibm * Cuu, cosPhibm * Cvv, cosPhibm * Cuv

def focal_position(nx, ny, x_h, y_h):
    nz = - sqrt(1 - nx**2 - ny**2)
    z_h = (x_h**2 + y_h**2 - 1)/2
    factor = sqrt(1 - ny**2)
    
    Ux = nz / factor
    Uy = 0
    Uz = -nx / factor

    Vx = ny * Uz
    Vy = nz * Ux - nx * Uz
    Vz = - ny * Ux

    cosPhi = 1/sqrt(1 + x_h**2 + y_h**2)

    r_h2 = x_h**2 + y_h**2

    nx_m = -x_h * cosPhi
    ny_m = -y_h * cosPhi
    nz_m = cosPhi

    cosPhibm = nx * nx_m + ny * ny_m + nz * nz_m


    na2 = (x_h * nx - y_h * ny - nz)**2
    Cuu = cosPhi * cosPhibm * (Vx**2 + Vy**2 + (2 * x_h * Vx + 2 * y_h * Vy + r_h2 * Vz) * Vz) / na2
    Cvv = cosPhi * cosPhibm * (Ux**2 + Uy**2 + (2 * x_h * Ux + 2 * y_h * Uy + r_h2 * Uz) * Uz) / na2
    Cuv = -cosPhi * cosPhibm * (Ux*Vx + Uy*Vy + x_h * (Ux*Vz+Vx*Uz) + y_h * (Uy*Vz+Vy*Uz) + r_h2 * Uz*Vz) / na2

    ndotnm = nx * nx_m + ny * ny_m + nz * nz_m
    nx_refl = nx - 2 * nx_m * ndotnm
    ny_refl = ny - 2 * ny_m * ndotnm
    nz_refl = nz - 2 * nz_m * ndotnm

    focal_strength = (Cuu + Cvv) / 2
    focal_length = -1 / (2*focal_strength)

    print(f'Cuu = {Cuu}, Cvv = {Cvv}')
    print(f'Focal Length = {focal_length}')
    print(f'Angles: {cosPhi}, {cosPhibm}')

    xf = x_h + nx_refl * focal_length
    yf = y_h + ny_refl * focal_length
    zf = z_h + nz_refl * focal_length
    return (xf, yf, zf), (nx_refl, ny_refl, nz_refl)

