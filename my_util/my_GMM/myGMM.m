classdef myGMM < handle
    %MYGMM self-defined GMM class for robot learning
    %   Detailed explanation goes here
    % Author: Tang Kai
    properties
        Mu; %k X dim
        Sigma; %dim X dim X k
        MixCoeffs; %mixture coefficients k X 1
        numGM; %number of gaussian models
        dimension; %data dimensions
        options_EM = struct("MinIterations", 5, "MaxIterations", 150, "FuncTol", 1e-4, ...
                                            "SigmaRegFac", 1e-8, "plotHistory", true);
    end

    methods
        function obj = myGMM(numGaussians)
            %MYGMM Construct an instance of this class
            %   Detailed explanation goes here
            obj.numGM = numGaussians;
        end

        function [Mu, Sigma, Coeffs] = initGMM_kmeans(obj, data)
            %INITGMM_KMEANS init GMM using K-means
            %   data: n X dim
            N = size(data, 1);
            obj.dimension = size(data, 2);
            obj.Sigma = zeros(obj.dimension, obj.dimension, obj.numGM);
            obj.MixCoeffs = zeros(obj.numGM, 1);

            [idx, Mu] = kmeans(data, obj.numGM); %K-means
            obj.Mu = Mu; 

            %Compute Covariance & Coefficients
            for i=1:obj.numGM
                clusterId_i = find(idx == i);
                numMembers = length(clusterId_i);
                members = data(clusterId_i, :);

                obj.MixCoeffs(i) = numMembers/N; %init Coefficients
                
                %init cov
                Mu_i = repmat(Mu(i, :), numMembers, 1);
                centralMembers = members - Mu_i;
                obj.Sigma(:, :, i) = 1/(numMembers-1) * (centralMembers') * centralMembers;
            end

            Coeffs =obj.MixCoeffs;
            Sigma = obj.Sigma;
        end
         
        function [likelihoodHistory, iters, exitFlag] = train_EM(obj, data, options)
            %TRAIN_EM train GMM using EM algorithm
            %   exitFlag: 0-reach maximum iterations; 1-converged
            if exist("options", "var")
                obj.options_EM = options;
            end
            
            N = size(data, 1);
            
            %init
            iters = 0;
            likelihoodHistory = []; %Record likelihood
            gaussProb = zeros(obj.numGM, N); %k X N gaussian PDF matrix
            diagReguTerm = eye(obj.dimension) * obj.options_EM.SigmaRegFac; % diagonal Regularization term for covariance matrix
            obj.Sigma = obj.Sigma +diagReguTerm; 

            while true
                %Compute gaussian PDF matrix
                for k=1:obj.numGM
                    gaussProb(k, :) = gaussianPDF(data', obj.Mu(k, :)', obj.Sigma(:, :, k)); %k X N
                end
                
                %Compute denominator of Gamma(n, k): 
                gammaDenom = sum(obj.MixCoeffs .* gaussProb, 1); % 1 X N
                
                %Likelihood Evaluation
                likelihood = sum(log(gammaDenom))/N; %normalized log likelihood

                %Record current Likelihood
                likelihoodHistory = [likelihoodHistory, likelihood];

                %Check stopping criterion
                if iters>=obj.options_EM.MaxIterations
                    exitFlag = 0;
                    disp("Reached maximum iterations!");
                    break;
                end

                if iters>=obj.options_EM.MinIterations && ...
                abs(likelihoodHistory(iters+1)-likelihoodHistory(iters))<=obj.options_EM.FuncTol
                    exitFlag = 1;
                    fprintf("EM converged after %d iterations.\n", iters);
                    break;
                end

                %Perform E-M
                for k=1:obj.numGM
                    %E-step (Gamma)
                     Gamma_k = obj.MixCoeffs(k) * (gaussProb(k, :) ./ gammaDenom); % 1 X N
                    %M-step
                    N_k = sum(Gamma_k);
                    obj.MixCoeffs(k) = N_k/N; %update mixture coefficients
                    obj.Mu(k, :) = sum(Gamma_k' .* data, 1) ./ N_k; %update expectation
                    obj.Sigma(:, :, k) = (Gamma_k .* (data - obj.Mu(k, :))') * (data - obj.Mu(k, :)) ./ N_k + diagReguTerm; %update covariance
                end

                iters = iters+1;

            end

            if obj.options_EM.plotHistory
                figure;
                plot(1:iters+1, likelihoodHistory);
                xlabel("Iterations"); ylabel("Log-Likelihood");
                title("Log Likelihood History");
            end
        end
        
        function  h = plotGMM2D(obj, ax, color, valAlpha)
            h = plotGMM2D(ax, obj.Mu, obj.Sigma, color, valAlpha);
        end

        function prob = evalPDF(obj, x)
            %EVALPDF evaluate PDF at point x
            %   x: dim X 1
            %Compute gaussian PDF
            gaussProb = zeros(obj.numGM, 1);
            for k=1:obj.numGM
                gaussProb(k) = gaussianPDF(x, obj.Mu(k, :)', obj.Sigma(:, :, k)); %k X 1
            end
            %Compute GMM PDF: 
            prob = obj.MixCoeffs' * gaussProb;
        end
    end
end