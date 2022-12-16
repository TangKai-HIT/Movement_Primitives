classdef TPGMM < handle
    %TPGMM task-parameterized GMM
    %   Detailed explanation goes here

    properties
        Mu; %1 X numFrames cell array, elements: k X dim
        Sigma; %1 X numFrames cell array, elements: dim X dim X k
        MixCoeffs; %mixture coefficients k X 1
        numGM; %number of gaussian models
        numFrames %number of task frames
        dimension; %data dimensions
        options_EM = struct("MinIterations", 5, "MaxIterations", 150, "FuncTol", 1e-4, ...
                                            "SigmaRegFac", 1e-8, "plotHistory", true);
    end

    methods
        function obj = TPGMM(numGM, numFrames, dim)
            %TPGMM Construct an instance of this class
            %   Detailed explanation goes here
            obj.numGM = numGM;
            obj.dimension = dim;
            obj.numFrames = numFrames;
        end

        function [Mu, Sigma] = retrieveTPGMM(obj, framesPose)
            %RETRIEVETPGMM reproduce new GMM for frames in new poses
            %   framesPose: 1 X numFrames struct array, members: A, b
            %   Mu: k X dim
            %   Sigma
            Mu = zeros(size(obj.Mu{1}))'; %dim X k
            Sigma = zeros(size(obj.Sigma{1}));
            for k=1:obj.numGM
                for m=1:obj.numFrames
                    xi_hat_m = framesPose(m).A * obj.Mu{m}(k, :)' + framesPose(m).b;
                    Sigma_hat_m = framesPose(m).A * obj.Sigma{m}(:, :, k) * framesPose(m).A';
                    invSigma_hat_m = inv(Sigma_hat_m);
                    Sigma(:, :, k) = Sigma(:, :, k) + invSigma_hat_m;
                    Mu(:, k) = Mu(:, k) + invSigma_hat_m * xi_hat_m;
                end
                Sigma(:, :, k) = inv(Sigma(:, :, k));
                Mu(:, k) =  Sigma(:, :, k) * Mu(:, k);
            end
            Mu = Mu'; %K X dim
        end
        
        function frameData = getFrameData(obj, rawData, framesPose)
            %RETRIEVETPGMM reproduce new GMM for frames in new poses
            %   rawData: 1 X demons cell array, {data: dim X N} 
            %   framesPose: 1 X demons struct array, members: A, b -- 1 X numframes cell array
            %   frameData: 1 X numframes cell array
            frameData = cell(1, obj.numFrames);
            numDemons = length(rawData);

            for i=1:obj.numFrames
                for m =1:numDemons
                    frameData{i} = [frameData{i} ;(framesPose(m).A{i}\rawData{m} - framesPose(m).b{i})'];
                end
            end
        end

        function initTPGMM_kmeans(obj, frameData)
            %INITTPGMM_KMEANS init GMM using K-means
            %   frameData: 1 x numframes cell array, {N X dim frame data} 
            N = size(frameData{1}, 1);
            obj.Sigma = cell(1, obj.numFrames);
            obj.Mu = cell(1, obj.numFrames);
            obj.MixCoeffs = zeros(obj.numGM, 1);

            %Init parameters & Concatenate data in different frames along dimension
            data = zeros(N, obj.numFrames * obj.dimension);
            for m=1:obj.numFrames
                obj.Mu{m} = zeros(obj.numGM, obj.dimension);
                obj.Sigma{m} = zeros(obj.dimension, obj.dimension, obj.numGM);

                startId = (m-1)*obj.dimension+1;
                endId = m*obj.dimension;
                data(:, startId : endId) = frameData{m};
            end

            [idx, concatMu] = kmeans(data, obj.numGM); %K-means

            %Compute Covariance & Coefficients
            for i=1:obj.numGM
                clusterId_i = find(idx == i);
                numMembers = length(clusterId_i);
                members = data(clusterId_i, :);

                obj.MixCoeffs(i) = numMembers/N; %init Coefficients
                
                %init  Mu, Sigma
                Mu_i = repmat(concatMu(i, :), numMembers, 1);
                centralMembers = members - Mu_i;

                for m=1:obj.numFrames
                    obj.Mu{m}(i, :) = concatMu(i, (m-1)*obj.dimension+1 : m*obj.dimension);
                    centralMembers_m = centralMembers(:, (m-1)*obj.dimension+1 : m*obj.dimension);
                    obj.Sigma{m}(:, :, i) = 1/(numMembers-1) * (centralMembers_m') * centralMembers_m;
                end
            end
        end

        function [likelihoodHistory, iters, exitFlag] = train_EM(obj, frameData, options)
            %TRAIN_EM train GMM using EM algorithm
            %   frameData: 1 x P cell array, {N X dim frame data} 
            %   exitFlag: 0-reach maximum iterations; 1-converged
            if exist("options", "var")
                obj.options_EM = options;
            end
            
            N = size(frameData{1}, 1);
            
            %init
            iters = 0;
            likelihoodHistory = []; %Record likelihood
            diagReguTerm = eye(obj.dimension) * obj.options_EM.SigmaRegFac; % diagonal Regularization term for covariance matrix

            while true
                %Compute gaussian PDF matrix
                gaussProb = ones(obj.numGM, N); %k X N gaussian PDF matrix
                for k=1:obj.numGM
                    for m=1:obj.numFrames
                        gaussProb(k, :) = gaussProb(k, :) .* gaussianPDF(frameData{m}', obj.Mu{m}(k, :)', obj.Sigma{m}(:, :, k)); %k X N
                    end
                end
                
                %Compute denominator of Gamma(n, k): 
                gammaDenom = sum(obj.MixCoeffs .* gaussProb, 1) + realmin; % 1 X N
                
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

                    for  m=1:obj.numFrames
                        obj.Mu{m}(k, :) = sum(Gamma_k' .* frameData{m}, 1) ./ N_k; %update expectation
                        centralData_m = frameData{m} - obj.Mu{m}(k, :);
                        obj.Sigma{m}(:, :, k) = (Gamma_k .* centralData_m') * centralData_m ./ N_k...
                                                            + diagReguTerm; %update covariance
                    end
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

    end
end