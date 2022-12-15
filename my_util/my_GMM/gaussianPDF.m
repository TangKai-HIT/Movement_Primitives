function prob = gaussianPDF(Data, Mu, Sigma)
% GAUSSIANPDF  Gaussian PDF evaluated at datapoint(s)
% Inputs -----------------------------------------------------------------
%   Data:  dim x N array representing N datapoints of D dimensions.
%   Mu:    dim x 1 vector representing the center of the Gaussian.
%   Sigma: dim x dim array representing the covariance matrix of the Gaussian.
% Output -----------------------------------------------------------------
%   prob:  1 x N vector representing the likelihood of the N datapoints.

[dim, N] = size(Data);
Data = Data - repmat(Mu,1,N);
prob = sum((Sigma\Data).*Data,1);
prob = exp(-0.5*prob) / sqrt((2*pi)^dim * abs(det(Sigma)) + realmin);