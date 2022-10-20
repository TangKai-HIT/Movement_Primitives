function psi = VonMises_Basis(Phi, c, sigma)
%VonMise_Basis: Von-Mises basis with 2 hyperparameters for periodic process
% Inputs -----------------------------------------------------------------
%   Phi:  phase (>0)
%   c:    center
%   sigma: l^2 or variance
% Output -----------------------------------------------------------------
%   psi
Phi = mod(Phi, 2*pi); %constrained to 0~2*pi
psi = exp(0.5* (cos(Phi-c) - 1) ./ sigma);
end

