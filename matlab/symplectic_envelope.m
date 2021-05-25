function [I, A_focus] = symplectic_envelope(A, x, y, N)
% Solve the focusing transverse envelope evolution equation using
% symplectic Euler. Needs small step size for stability, so if it blows up
% increase N. 

R2 = x.^2 + y.^2;
laplacian = @(A) (circshift(A, -1, 1) - 2*A + circshift(A, 1, 1))/mean(diff(x))^2 ...
               + (circshift(A, -1, 2) - 2*A + circshift(A, 1, 2))/mean(diff(y))^2;

Ar = real(A);
Ai = imag(A);
dt = pi/N;
middle = floor(size(A, 2) / 2) + 1;
I = abs(A(:,middle)).^2;
Nplot = 100;
for i = 1:N
Ai = Ai - dt * ( laplacian(Ar) + 4*(1 - R2) .* Ar ) / 4;
Ar = Ar + dt * ( laplacian(Ai) + 4*(1 - R2) .* Ai ) / 4;
if mod(i, Nplot) == 0
    II = Ar.^2 + Ai.^2;
    I = [I, II(:,middle)];
    cla;
    imagesc(II);
    axis equal tight;
    title(sprintf('guoy angle %f pi', i/N-0.5));
    colorbar;
    drawnow;
end
if i == floor(N/2)
    A_focus = Ar + 1j * Ai;
end
end


end

