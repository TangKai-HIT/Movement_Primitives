function h = plotGMM2D(ax, Mu, Sigma, color, valAlpha)
% This function displays the parameters of a Gaussian Mixture Model (GMM)
% Inputs -----------------------------------------------------------------
%   ax
%   Mu:           K x dim array representing the centers of K Gaussians.
%   Sigma:        dim x dim x K array representing the covariance matrices of K Gaussians.
%   color:        3 x 1 array representing the RGB color to use for the display.
%   valAlpha:     transparency factor (optional).

hold(ax, "on");

numGauss = size(Mu,1);
nbDrawingSeg = 100;
darkcolor = color * .7; %max(color-0.5,0);
t = linspace(-pi, pi, nbDrawingSeg);
if nargin<5
	valAlpha = 1;
end

h = [];
X = zeros(2,nbDrawingSeg,numGauss);
for i=1:numGauss
	[V,D] = eig(Sigma(:,:,i));
	R = real(V*D.^.5);
% 	R = chol(Sigma(:,:,i))';
% 	R = sqrtm(Sigma(:,:,i));
	X(:,:,i) = R * [cos(t); sin(t)] + repmat(Mu(i,:)', 1, nbDrawingSeg);
	if nargin>4 %Plot with alpha transparency
		h = [h patch(ax, X(1,:,i), X(2,:,i), color, 'lineWidth', 1, 'EdgeColor', color, 'facealpha', valAlpha,'edgealpha', valAlpha)];
		%MuTmp = [cos(t); sin(t)] * 0.3 + repmat(Mu(i,:)',1,nbDrawingSeg);
		%h = [h patch(ax, MuTmp(1,:), MuTmp(2,:), darkcolor, 'LineStyle', 'none', 'facealpha', valAlpha)];
		h = [h plot(ax, Mu(:,1), Mu(:,2), '.', 'markersize', 10, 'color', darkcolor)];
	else %Plot without transparency
		%Standard plot
		h = [h patch(ax, X(1,:,i), X(2,:,i), color, 'lineWidth', 1, 'EdgeColor', darkcolor)];
		h = [h plot(ax, Mu(:,1), Mu(:,2), '.', 'markersize', 10, 'color', darkcolor)];
% 		%Plot only contours
% 		h = [h plot(ax, X(1,:,i), X(2,:,i), '-', 'color', color, 'lineWidth', 1)];
	end
end