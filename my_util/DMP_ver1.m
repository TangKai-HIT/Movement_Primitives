classdef DMP_ver1 < handle
    %DMP_ver1: Class of general(discrete + periodic) DMP trained by Recursive LWR  or GMR
    %   Describtion-------------------------------------
    %   
    
    properties
        DMP_Params=struct('goal', [], 'y_0', [], 'dy_0', [], 'alpha_z', 25, 'beta_z', 25/4, ...
                                        'alpha_x', 25/3, 'tau', [], 'canonVarDim', 1, 'transVarDim', []);
                                    
        Force_Params=struct('nbFuncs', [], 'weights', [], 'Mu', [], 'Sigma', []);
        
        Trajectory;
        
        TrainData = {};
        
        TrainMethod;
        LWR_polyOrder;
        
        nbDemons = 0; %number of demonstrations
        dt =0.01;
    end
    
    properties (Access = private)
        TrainDataTemplate = struct('y_train', [], 'dy_train', [], 'ddy_train', [], 'nbData', []); %template of each demonstration data
        TrajectoryTemplate = struct('timeSq', [], 's_traj', [], 'y_traj', [], 'dy_traj', [], 'ddy_traj', [], 'f_traj', [], 'activations', []); %template of query trajectory data
        
        lastDataId_rlwr = 0;  %last trained index of demonstration data using RLWR
        P_rlwr; %P matrix for recursive LWR
        lambda=1; %forgetting factor
    end
    
    methods
        %% DMP_ver1: Constructor-------------------------------------------------------------------
        function obj = DMP_ver1(alpha_z, beta_z, tau, alpha_x, dt, ...
                                            trainPosData, trainVelData, trainAccelData)
            %DMP_ver1: Construct an instance of this class
            %   inputs: trainPosData--cell array containing demonstrations
            %   position data Yi (Yi: d X N matrix)
            if nargin>0
                %%specify parameters of canonical & transformation systems
                obj.DMP_Params.alpha_z = alpha_z;
                obj.DMP_Params.beta_z = beta_z;
                obj.DMP_Params.tau = tau;
                obj.DMP_Params.alpha_x = alpha_x;
%                 obj.DMP_Params.canonVarDim = canonVarDim; %dimension of canonical variable x (or s)

                %%specify time increment
                obj.dt = dt;
                
                if nargin>5 % if input training data
                    %%specify training data(in batch)
                    if nargin>7 %input all parameters
                    	obj.inputNewDemons(trainPosData, trainVelData, trainAccelData);
                    elseif exist('trainVelData','var')
                        obj.inputNewDemons(trainPosData, trainVelData);
                    else
                        obj.inputNewDemons(trainPosData);
                    end
                end
                
            end
        end
        
        %% inputNewDemons--------------------------------------------------------------------------
        function inputNewDemons(obj, trainPosData, trainVelData, trainAccelData)
            %inputNewDemons: input new demonstrations in batch
            %   inputs:  trainPosData--batches of demonstrated position data(DXN) stored in cell array    
            
            if ~iscell(trainPosData) %if input single demonstration data in array
                trainPosData = {trainPosData};
            end
            %update number of demonstrations
            nbDemons_old = obj.nbDemons; %record last index of demonstration
            obj.nbDemons = obj.nbDemons + size(trainPosData,2);

            for i = nbDemons_old+1 : obj.nbDemons
                obj.TrainData{i} = obj.TrainDataTemplate; %assign empty template
                %obj.TrainData{i}.y_train = spline(1:size(trainPosData{i},2), trainPosData{i}, linspace(1,size(trainPosData{i},2),nbSampData)); %Resampling
                obj.TrainData{i}.y_train = trainPosData{i};
                obj.TrainData{i}.nbData = size(trainPosData{i}, 2);
                
                if ~exist('trainVelData','var')
                    obj.TrainData{i}.dy_train = gradient(trainPosData{i})/obj.dt;
                else
                    if ~iscell(trainVelData) %if input single demonstration data in array
                        trainVelData = {trainVelData};
                    end
                    obj.TrainData{i}.dy_train = trainVelData{i};
                end
                
                if ~exist('trainAccelData','var')
                    obj.TrainData{i}.ddy_train = gradient(obj.TrainData{i}.dy_train)/obj.dt;
                else
                    if ~iscell(trainAccelData) %if input single demonstration data in array
                        trainAccelData = {trainAccelData};
                    end
                    obj.TrainData{i}.ddy_train = trainAccelData{i};
                end
                
            end
            
            %dimension of ouput variable y
            if isempty(obj.DMP_Params.transVarDim)
                obj.DMP_Params.transVarDim = size(obj.TrainData{1}.y_train, 1); 
            end
            %set default goal & initial states
            if isempty(obj.DMP_Params.goal)
                obj.DMP_Params.goal = obj.TrainData{1}.y_train(:, end); %set default goal
            end
            if isempty(obj.DMP_Params.y_0)
                obj.DMP_Params.y_0 = obj.TrainData{1}.y_train(:, 1); %set default initial state
            end
            if isempty(obj.DMP_Params.dy_0)
                obj.DMP_Params.dy_0 = obj.TrainData{1}.dy_train(:, 1); %set default initial state
            end
        end
        
        %% init_RBFBasis_timeBased ---------------------------------------------
        function init_RBFBasis_timeBased(obj, nbFuncs)
            %init_RBFBasis_timeBased1: init RBF basis functions to evenly distributed on time interval.
            %   Inputs: nbFuncs--number of basis functions
            
            obj.Force_Params.Mu = zeros(1,nbFuncs);
            obj.Force_Params.Sigma = zeros(1,nbFuncs);
            obj.Force_Params.nbFuncs =nbFuncs;
            
            for i=1:nbFuncs
                obj.Force_Params.Mu(i) = exp(-obj.DMP_Params.alpha_x / obj.DMP_Params.tau *(i-1)/(nbFuncs-1));
                if i > 1
                    obj.Force_Params.Sigma(i-1) = ((obj.Force_Params.Mu(i) - obj.Force_Params.Mu(i-1))/2)^2;
                end
            end
            obj.Force_Params.Sigma(nbFuncs) = obj.Force_Params.Sigma(nbFuncs-1);
            
            obj.Force_Params.weights = ones(obj.DMP_Params.transVarDim, nbFuncs) * (1/nbFuncs);
        end
        
        %% init_RBFBasis_stateBased ---------------------------------------------
        function init_RBFBasis_stateBased(obj, nbFuncs)
            %init_RBFBasis_stateBased: init RBF basis functions to clusters evenly distributed on canonical states interval (0~1).
            %   Inputs: nbFuncs--number of basis functions
            
            params_diagRegFact = 1E-4; %Optional regularization term to avoid numerical instability
            obj.Force_Params.Mu = zeros(1,nbFuncs);
            obj.Force_Params.Sigma = zeros(1,nbFuncs);
            obj.Force_Params.nbFuncs =nbFuncs;
            
            if isempty(obj.TrainData)
                disp("Please input training data first!");
                return
            else
                maxNumData=zeros(1, size(obj.TrainData, 2));
                for i=1:size(obj.TrainData, 2)
                    maxNumData(i) = obj.TrainData{i}.nbData;
                end
                maxNumData = max(maxNumData);
            end
            
            time = obj.dt * (0 : maxNumData);
            S = obj.genCanonStates(time);
            
            TimingSep = linspace(min(S(1,:)), max(S(1,:)), nbFuncs+1);
            
            obj.Force_Params.weights = zeros(obj.DMP_Params.transVarDim, nbFuncs);
            for i=1:nbFuncs
                idtmp = find( S>=TimingSep(i) & S<TimingSep(i+1));
                obj.Force_Params.weights(i) = length(idtmp);
                obj.Force_Params.Mu(i) = mean(S(idtmp));
                obj.Force_Params.Sigma(i) = cov(S(idtmp));
                %Optional regularization term to avoid numerical instability
                obj.Force_Params.Sigma(i) = obj.Force_Params.Sigma(i) + params_diagRegFact;
            end
            
            obj.Force_Params.weights = obj.Force_Params.weights ./ sum(obj.Force_Params.weights, 2);
        end
        
        %% genCanonStates----------------------------------------------------------------------
        function s_Query=genCanonStates(obj, timeQuery)
            %genCanonStates: generate canonical system states by query timing points(i.e. trajectory of x)
            %   inputs:
            s_Query = exp(-obj.DMP_Params.alpha_x/obj.DMP_Params.tau*timeQuery); 
        end
        
        %% genPredTraj_LWR--------------------------------------------------------------------------
        function trajQuery=genPredTraj_LWR(obj, endTime)
            %genPredTraj_LWR: generate predicted output states of LWR-trained model by query timing points(i.e. trajectory of y)
            %   inputs:
            
            trajQuery  = obj.TrajectoryTemplate; %init using trajectory template structure
            %generate canonical states
            trajQuery.timeQuery = 0: obj.dt: endTime;
            trajQuery.s_traj = obj.genCanonStates(trajQuery.timeQuery); 
            %compute Phi matrix(basis functions)
            Phi = zeros(obj.Force_Params.nbFuncs, size(trajQuery.timeQuery,2));
            for k =1:obj.Force_Params.nbFuncs
                Phi(k, :) = rbf_Basis(trajQuery.s_traj, obj.Force_Params.Mu(k), obj.Force_Params.Sigma(k));
            end
            trajQuery.activations = Phi; %record activations
            Phi = Phi ./ sum(Phi,1);
            %Mapping the input states to polynomial space
            poly_S = zeros(obj.LWR_polyOrder+1, size(trajQuery.s_traj,2));
            for n=0:obj.LWR_polyOrder
                poly_S(n+1, :) = trajQuery.s_traj.^n;
            end
            
            %compute force terms
            y_pred = zeros(obj.DMP_Params.transVarDim, size(poly_S, 2), obj.Force_Params.nbFuncs); %D x N x k
            for i=1:obj.Force_Params.nbFuncs
                y_pred(:, :, i) = obj.Force_Params.weights(:, :, i) * poly_S;
            end
            
            for i=1:obj.DMP_Params.transVarDim
                temp = reshape(y_pred(i, :, :), size(poly_S, 2), obj.Force_Params.nbFuncs);
                trajQuery.f_traj(i, :) = sum(temp' .* Phi, 1);
            end
            
            %Reproduction: Eular forward iteration
            N = size(trajQuery.timeQuery,2); %number of query points
            trajQuery.y_traj = zeros(obj.DMP_Params.transVarDim, N);
            trajQuery.dy_traj = trajQuery.y_traj;
            trajQuery.ddy_traj = trajQuery.y_traj;
            
            for i=1:N
                if i==1
                    trajQuery.y_traj(:, i) = obj.DMP_Params.y_0;
                    trajQuery.dy_traj(:, i) = obj.DMP_Params.dy_0;
                else
                    trajQuery.y_traj(:, i) =  trajQuery.y_traj(:, i-1) +  trajQuery.dy_traj(:, i-1) * obj.dt;
                    trajQuery.dy_traj(:, i) = trajQuery.dy_traj(:, i-1) +  trajQuery.ddy_traj(:, i-1) * obj.dt;
                end

                trajQuery.ddy_traj(:, i) = 1/obj.DMP_Params.tau^2 * (trajQuery.f_traj(:, i) +  obj.DMP_Params.alpha_z * ...
                      (obj.DMP_Params.beta_z*(obj.DMP_Params.goal - trajQuery.y_traj(:, i)) - obj.DMP_Params.tau * trajQuery.dy_traj(:, i)));
            end
            
            obj.Trajectory = trajQuery;
        end 
        
        %% plot_Results-------------------------------------------------------------------------------
        function plot_Results(obj, saveFigName)
            %plot_Results: plot all figures including demonstrations, predicted
            %   trajectories, force term, activated basis functions
            %Init figures and plot parameters
            figure('PaperPosition',[0 0 16 8],'position',[50,80,1600,900],'color',[1 1 1]); 
            xx = round(linspace(1, 64, obj.Force_Params.nbFuncs)); %index to divide colormap
            clrmap = colormap('jet')*0.5;
            clrmap = min(clrmap(xx,:),.9);

            %Plot spatial demonstrations and predicted trajectory
            axes('Position',[0 0 .2 1]); hold on; axis off;
            for i=1:size(obj.TrainData, 2)
            plot(obj.TrainData{i}.y_train(1,:), obj.TrainData{i}.y_train(2,:), '.', 'markersize', 8, 'color', [.7 .7 .7]);
            end
            plot(obj.Trajectory.y_traj(1,:), obj.Trajectory.y_traj(2,:), '-', 'linewidth', 3, 'color', [.8 0 0]);
            axis equal; axis square; 
            if obj.TrainMethod == 'RLWR'
                title(sprintf("Trained By recursive-LWR\n$\\lambda=%.2f$\nPolynomial order: %d", obj.lambda, obj.LWR_polyOrder), ...
                                'fontsize',16,'interpreter','latex');
            elseif obj.TrainMethod == 'BLWR'
                title(sprintf("Trained By batch-LWR\nPolynomial order: %d", obj.LWR_polyOrder), 'fontsize',16,'interpreter','latex');
            end
            
            %Timeline plot of the force term
            axes('Position',[.25 .58 .7 .4]); hold on; 
            plot(obj.Trajectory.timeQuery, obj.Trajectory.f_traj(1,:), '-','linewidth', 2, 'color', [.8 0 0]);
            plot(obj.Trajectory.timeQuery, obj.Trajectory.f_traj(2,:), '-','linewidth', 2, 'color', [.5 0 0]);
            % axis([min(timeSqe) max(timeSqe) min(trajQuery.f_traj(1,:)) max(trajQuery.f_traj(1,:))]);
            legend('$f_1(x)$','$f_2(x)$','fontsize',12,'interpreter','latex')
            ylabel('$Force$','fontsize',16,'interpreter','latex');
            xlabel('$t/s$','fontsize',16,'interpreter','latex');
            view(180,-90);

            %Plot of the basis functions activation w.r.t canonical state
            axes('Position',[.25 .12 .7 .4]); hold on; 
            for i=1:obj.Force_Params.nbFuncs
                patch([obj.Trajectory.s_traj(1), obj.Trajectory.s_traj, obj.Trajectory.s_traj(end)], ...
                            [0, obj.Trajectory.activations(i,:), 0], min(clrmap(i,:)+0.5,1), 'EdgeColor', 'none', 'facealpha', .4);
                plot(obj.Trajectory.s_traj, obj.Trajectory.activations(i,:), 'linewidth', 2, 'color', min(clrmap(i,:)+0.2,1));
            end
            % axis([min(sIn) max(sIn) 0 1]);
            xlabel('$x$','fontsize',16,'interpreter','latex'); 
            ylabel('$\Psi$','fontsize',16,'interpreter','latex');
            view(180,-90);

            %Save figures
            if exist('saveFig', 'var')
                for i = 1:size(saveFigName, 2)
                    saveas(gcf, saveFigName{i});
                end
            end
        end
        
        %% LWR_batchTrain-----------------------------------------------------------------------------------
        function LWR_batchTrain(obj)
            %LWR_batchTrain: train the DMP with batch locally weighted
            %    regression,Local regression type: linear
            obj.TrainMethod = 'BLWR';
            
            if isempty(obj.LWR_polyOrder)
                obj.LWR_polyOrder = 1;
                disp("LWR_polyOrder empty! Set LWR_polyOrder to default 1.")
            end
            
            %%construct weight matrix Phi , input vector S and compute Force term
            S=[];
            W = [];
            F_d = [];
            for i=1:obj.nbDemons
                %canonical states for current demonstration
                s = obj.genCanonStates([0 : obj.TrainData{i}.nbData-1]*obj.dt);
                S = [S, s];
                %Compute weights of Input demonstrated data
                w = zeros(obj.Force_Params.nbFuncs, obj.TrainData{i}.nbData);
                for k =1:obj.Force_Params.nbFuncs
                    w(k, :) = rbf_Basis(s, obj.Force_Params.Mu(k), obj.Force_Params.Sigma(k));
                end
                w = w ./ sum(w,1);
                W = [W, w]; %compress diagonal weight matrix W to an array
                %Output force term data
                g = obj.TrainData{i}.y_train(:, end);
                f_d = obj.DMP_Params.tau^2 * obj.TrainData{i}.ddy_train - ...
                    obj.DMP_Params.alpha_z*(obj.DMP_Params.beta_z*(g - obj.TrainData{i}.y_train) - obj.DMP_Params.tau*obj.TrainData{i}.dy_train);
                F_d = [F_d, f_d];
            end
            
            %Mapping the input states to polynomial space
            poly_S = zeros(obj.LWR_polyOrder+1, size(S,2));
            for n=0:obj.LWR_polyOrder
                poly_S(n+1, :) = S.^n;
            end
            
            %solution of weights
            obj.Force_Params.weights = zeros(obj.DMP_Params.transVarDim, obj.LWR_polyOrder+1, obj.Force_Params.nbFuncs); %D X n X k 
            for i=1:obj.Force_Params.nbFuncs
                obj.Force_Params.weights(:, :, i) = (F_d.*W(i, :)) * poly_S' / ((poly_S.*W(i, :)) * poly_S');
            end
            
        end
        
          %% RLWR_Train---------------------------------------------------------------------------
        function RLWR_Train(obj, lambda)
            %LWR_batchTrain: train the DMP with recursive locally weighted
            %regression
            obj.TrainMethod = 'RLWR';
            if isempty(obj.LWR_polyOrder)
                obj.LWR_polyOrder = 1;
                disp("LWR_polyOrder empty! Set LWR_polyOrder to default 1.")
            end
            
            if exist('lambda', 'var')
                obj.lambda = lambda;
            end
            
            if isempty(obj.P_rlwr) %use RLWR for the first time
                obj.Force_Params.weights = zeros(obj.DMP_Params.transVarDim, obj.LWR_polyOrder+1, obj.Force_Params.nbFuncs); %D X n X k 
                P_dim = obj.LWR_polyOrder + 1;
                obj.P_rlwr = 10*ones(P_dim, P_dim, obj.Force_Params.nbFuncs) .* eye(P_dim); %init P matrix
            end
            
            if obj.lastDataId_rlwr == obj.nbDemons
                disp('Please input new data!');
            else
                %%construct input data
                W = [];
                F_d = [];
                S = [];
                for id = obj.lastDataId_rlwr+1 : obj.nbDemons
                    %canonical states for current demonstration
                    s = obj.genCanonStates([0 : obj.TrainData{id}.nbData-1]*obj.dt);
                    S = [S, s];
                    %Input demonstrated data
                    w = zeros(obj.Force_Params.nbFuncs, obj.TrainData{id}.nbData);
                    for k =1:obj.Force_Params.nbFuncs
                        w(k, :) = rbf_Basis(s, obj.Force_Params.Mu(k), obj.Force_Params.Sigma(k));
                    end
                    w = w ./ sum(w,1);
                    W = [W, w];
                    %Output force term data
                    g = obj.TrainData{id}.y_train(:, end);
                    f_d = obj.DMP_Params.tau^2 * obj.TrainData{id}.ddy_train - ...
                        obj.DMP_Params.alpha_z*(obj.DMP_Params.beta_z*(g - obj.TrainData{id}.y_train) - obj.DMP_Params.tau*obj.TrainData{id}.dy_train);
                    F_d = [F_d, f_d];
                end
                
                %Mapping the input states to polynomial space
                poly_S = zeros(obj.LWR_polyOrder+1, size(S,2));
                for n=0:obj.LWR_polyOrder
                    poly_S(n+1, :) = S.^n;
                end
                
                %%recursively update the weights
                w_old = obj.Force_Params.weights; %D X n X k matrix
                P_old = obj.P_rlwr;
                for i = 1:size(F_d,2)
                    w = W(:, i);
                    xi = poly_S(:, i);
                    for k=1:obj.Force_Params.nbFuncs
                        w_k = w_old(:, :, k)';
                        obj.P_rlwr(:, :, k) = 1/lambda * (P_old(:, :, k) - (P_old(:, :, k)*(xi * xi' )*P_old(:, :, k))/(lambda / w(k) + xi' * P_old(:, :, k) * xi));
                        error_k = F_d(:, i) - w_k' * xi;
                        obj.Force_Params.weights(:, :, k) = ( w_k + w(k) * (obj.P_rlwr(:, :, k) * xi) * (error_k') )';
                    end
                    w_old = obj.Force_Params.weights;
                    P_old = obj.P_rlwr;
                end
                obj.lastDataId_rlwr = obj.nbDemons;
                
            end
        end
        
    end
end

