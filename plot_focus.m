function plot_focus(x, y, A, W, focal_distance, lambda, Nstep)
% x: vector, shape (n_x, 1) in your units of choice (e.g. cm)
% y: vector, shape (1, n_y) in your units of choice
% A: matrix, shape (n_y, n_x) the complex e-field you want to focus
% W: scalar, the `radius' of A, in the same units as x and y
% focal_distance: the focal length of the optic, same units as w
% lambda: wavelength of the laser.
% Nstep: Number of steps to take in the Guoy phase angle

% If lambda is given in different units to the macroscopic distances
% x,y,W,focal_distance then the results will be plotted in the units of 
% lambda


x0 = x/W;
y0 = y/W;
[I, A_focus] = symplectic_envelope(A, x0, y0, Nstep);

guoy = linspace(-pi/2, pi/2, size(I, 2));
w0 = lambda * focal_distance / W / pi;
zR = pi * w0^2 / lambda;
z0 = tan(guoy);
w_of_z = w0 ./ cos(guoy);

I_focus = abs(A_focus).^2 * (W/w0)^2;

subplot(1,2,1);
imagesc(x0 * w0, y0 * w0, I_focus);
axis equal tight;

subplot(1,2,2);
surf(z0 * zR + 0*x0, x0 * w_of_z, I .* (W ./ w_of_z).^2)
view(2);
shading interp
xlim([-1, 1] * 10*zR); ylim([-1, 1] * 10*w0);
