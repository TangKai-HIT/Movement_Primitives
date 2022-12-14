function model = EM_tensorMFA(Data, model)
% Training of a task-parameterized mixture of factor analyzers (TP-MFA) with an 
% expectation-maximization (EM) algorithm.
%
% Writing code takes time. Polishing it and making it available to others takes longer! 
% If some parts of the code were useful for your research of for a better understanding 
% of the algorithms, please reward the authors by citing the related publications, 
% and consider making your own research available in this way.
%
% @article{Calinon16JIST,
%   author="Calinon, S.",
%   title="A Tutorial on Task-Parameterized Movement Learning and Retrieval",
%   journal="Intelligent Service Robotics",
%   publisher="Springer Berlin Heidelberg",
%   doi="10.1007/s11370-015-0187-9",
%   year="2016",
%   volume="9",
%   number="1",
%   pages="1--29"
% }
%
% Copyright (c) 2015 Idiap Research Institute, http://idiap.ch/
% Written by Sylvain Calinon, http://calinon.ch/
% 
% This file is part of PbDlib, http://www.idiap.ch/software/pbdlib/
% 
% PbDlib is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License version 3 as
% published by the Free Software Foundation.
% 
% PbDlib is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with PbDlib. If not, see <http://www.gnu.org/licenses/>.


%Parameters of the EM algorithm
nbMinSteps = 5; %Minimum number of iterations allowed
nbMaxSteps = 100; %Maximum number of iterations allowed
maxDiffLL = 1E-5; %Likelihood increase threshold to stop the algorithm
nbData = size(Data,3);

%diagRegularizationFactor = 1E-2; %Optional regularization term
diagRegularizationFactor = 1E-10; %Optional regularization term

% %Initialization of the MFA parameters
% Itmp = eye(model.nbVar)*1E-2;
% model.P = repmat(Itmp, [1 1 model.nbFrames model.nbStates]);
% model.L = repmat(Itmp(:,1:model.nbFA), [1 1 model.nbFrames model.nbStates]);

%Initialization of the MFA parameters
for i=1:model.nbStates
	for m=1:model.nbFrames
		model.P(:,:,m,i) = diag(diag(model.Sigma(:,:,m,i)));
		[V,D] = eig(model.Sigma(:,:,m,i)-model.P(:,:,m,i)); 
		[~,id] = sort(diag(D),'descend');
		V = V(:,id)*D(id,id).^.5;
		model.L(:,:,m,i) = V(:,1:model.nbFA);
	end
end
for nbIter=1:nbMaxSteps
	for i=1:model.nbStates
		for m=1:model.nbFrames
			%Update B,L,P
			B(:,:,m,i) = model.L(:,:,m,i)' / (model.L(:,:,m,i) * model.L(:,:,m,i)' + model.P(:,:,m,i));
			model.L(:,:,m,i) = model.Sigma(:,:,m,i) * B(:,:,m,i)' / (eye(model.nbFA) - B(:,:,m,i) * model.L(:,:,m,i) + B(:,:,m,i) * model.Sigma(:,:,m,i) * B(:,:,m,i)');
			model.P(:,:,m,i) = diag(diag(model.Sigma(:,:,m,i) - model.L(:,:,m,i) * B(:,:,m,i) * model.Sigma(:,:,m,i)));
		end
	end
end

%EM loop
for nbIter=1:nbMaxSteps
	fprintf('.');
	
	%E-step
	[Lik, GAMMA] = computeGamma(Data, model); %See 'computeGamma' function below
	GAMMA2 = GAMMA ./ repmat(sum(GAMMA,2),1,nbData);
	model.Pix = GAMMA2;
	
	%M-step
	for i=1:model.nbStates
		
		%Update Priors
		model.Priors(i) = sum(sum(GAMMA(i,:))) / nbData;
		
		for m=1:model.nbFrames
			%Matricization/flattening of tensor
			DataMat(:,:) = Data(:,m,:);
			
			%Update Mu
			model.Mu(:,m,i) = DataMat * GAMMA2(i,:)';
			
			%Compute covariance
			DataTmp = DataMat - repmat(model.Mu(:,m,i),1,nbData);
			S(:,:,m,i) = DataTmp * diag(GAMMA2(i,:)) * DataTmp' + eye(model.nbVar)*diagRegularizationFactor;
			
			%Update B
			B(:,:,m,i) = model.L(:,:,m,i)' / (model.L(:,:,m,i) * model.L(:,:,m,i)' + model.P(:,:,m,i));
			%Update Lambda
			model.L(:,:,m,i) = S(:,:,m,i) * B(:,:,m,i)' / (eye(model.nbFA) - B(:,:,m,i) * model.L(:,:,m,i) + B(:,:,m,i) * S(:,:,m,i) * B(:,:,m,i)');
			%Update Psi
			model.P(:,:,m,i) = diag(diag(S(:,:,m,i) - model.L(:,:,m,i) * B(:,:,m,i) * S(:,:,m,i))) + eye(model.nbVar)*diagRegularizationFactor;

			%Reconstruct Sigma
			model.Sigma(:,:,m,i) = model.L(:,:,m,i) * model.L(:,:,m,i)' + model.P(:,:,m,i);
		end
	end
	
	%Compute average log-likelihood
	LL(nbIter) = sum(log(sum(Lik,1))) / size(Lik,2);
	%Stop the algorithm if EM converged (small change of LL)
	if nbIter>nbMinSteps
		if LL(nbIter)-LL(nbIter-1)<maxDiffLL || nbIter==nbMaxSteps-1
			disp(['EM converged after ' num2str(nbIter) ' iterations.']);
			return;
		end
	end
end
disp(['The maximum number of ' num2str(nbMaxSteps) ' EM iterations has been reached.']);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Lik, GAMMA, GAMMA0] = computeGamma(Data, model)
nbData = size(Data, 3);
Lik = ones(model.nbStates, nbData);
GAMMA0 = zeros(model.nbStates, model.nbFrames, nbData);
for i=1:model.nbStates
	for m=1:model.nbFrames
		DataMat(:,:) = Data(:,m,:); %Matricization/flattening of tensor
		GAMMA0(i,m,:) = gaussPDF(DataMat, model.Mu(:,m,i), model.Sigma(:,:,m,i));
		Lik(i,:) = Lik(i,:) .* squeeze(GAMMA0(i,m,:))';
	end
	Lik(i,:) = Lik(i,:) * model.Priors(i);
end
GAMMA = Lik ./ repmat(sum(Lik,1)+realmin, size(Lik,1), 1);
end
