function psi = rbf_Basis(Data, Mu, Cov)
%rbf_Basis: multi-dimensional input RBF basis function with only 2 hyperparameters
% Inputs -----------------------------------------------------------------
%   Data:  D x N array representing N datapoints of D dimensions.
%   Mu:    D x 1 vector representing the center of the Gaussian.
%   Sigma: D x D array representing the covariance matrix of the Gaussian.
% Output -----------------------------------------------------------------
%   psi:  1 x N vector.
nbData = size(Data,2);
Data = Data - repmat(Mu,1,nbData);
psi = sum((Cov\Data).*Data,1);
psi = exp(-0.5*psi);
end

