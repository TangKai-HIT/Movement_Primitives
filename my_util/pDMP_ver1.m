 classdef pDMP_ver1 < handle
    %pDMP_ver1: Class of periodic DMP trained by Recursive LWR  or GMR, with
    %shared 1-dimensional canonical system (a phase oscillator phi)
    %   Describtion-------------------------------------
    %   : 1.modified from DMP_ver1 into periodic form;
    %     2.only state-based init function;
    
    properties
        DMP_Params=struct('goal', [], 'y_0', [], 'dy_0', [], 'alpha_z', [], 'beta_z', [], ...
                                        'tau', [], 'r', [], 'canonVarDim', 1, 'transVarDim', []);
                                    
        Force_Params=struct('nbFuncs', [], 'weights', [], 'Mu', [], 'Sigma', []);
        
        Trajectory;
        
        TrainData = {};
          
        lambda; %forgetting factor
        
        dt;
    end
    
    properties (SetAccess = private)
        TrainDataTemplate = struct('y_train', [], 'dy_train', [], 'ddy_train', [], 'nbData', []); %template of each demonstration data
        TrajectoryTemplate = struct('timeQuery', [], 's_traj', [], 'y_traj', [], 'dy_traj', [], 'ddy_traj', [], 'f_traj', [], 'activations', []); %template of query trajectory data
        
        nbDemons = 0; %number of demonstrations
        
        TrainMethod;
        
        lastDataId_rlwr = 0;  %last trained index of demonstration data using RLWR
        P_rlwr; %P matrix for recursive LWR
        LWR_polyOrder;
    end
    
    methods
        %% pDMP_ver1: Constructor-------------------------------------------------------------------
        function obj = pDMP_ver1(alpha_z, beta_z, tau, goal, r, dt, ...
                                            trainPosData, trainVelData, trainAccelData)
            %pDMP_ver1: Construct an instance of this class
            %   inputs: alpha_z--d X 1 array; trainPosData--cell array containing demonstrations
            %   position data Yi (Yi: d X N matrix)
            if nargin>0
                %%specify parameters of canonical & transformation systems
                obj.DMP_Params.alpha_z = alpha_z;
                obj.DMP_Params.beta_z = beta_z;
                obj.DMP_Params.tau = tau;
                obj.DMP_Params.goal = goal;
                obj.DMP_Params.r = r;
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
        
        %% init_Basis_stateBased ---------------------------------------------
        function init_Basis_stateBased(obj, nbFuncs)
            %init_Basis_stateBased: init Von-Mise basis functions to clusters evenly distributed on canonical states interval (0~1).
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
            S = mod(obj.genCanonStates(time), 2*pi); %constrained to [0, 2pi]
            
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
            %genCanonStates: generate canonical system states by query timing points(i.e. phase of phi)
            %   inputs:
            s_Query = timeQuery / obj.DMP_Params.tau; 
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
                Phi(k, :) = VonMises_Basis(trajQuery.s_traj, obj.Force_Params.Mu(k), obj.Force_Params.Sigma(k)); %use Von-Mises basis
            end
            trajQuery.activations = Phi; %record activations
            Phi = Phi ./ sum(Phi,1);
            %Mapping the input states to polynomial space
            poly_S = zeros(obj.LWR_polyOrder+1, size(trajQuery.s_traj,2));
            for n=0:obj.LWR_polyOrder
                poly_S(n+1, :) = mod(trajQuery.s_traj, 2*pi).^n; %constrained states to 0~2pi
            end
            
            %compute force terms
            y_pred = zeros(obj.DMP_Params.transVarDim, size(poly_S, 2), obj.Force_Params.nbFuncs); %D x N x k
            for i=1:obj.Force_Params.nbFuncs
                y_pred(:, :, i) = obj.Force_Params.weights(:, :, i) * poly_S * obj.DMP_Params.r; 
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

                trajQuery.ddy_traj(:, i) = 1./obj.DMP_Params.tau.^2 .* (trajQuery.f_traj(:, i) +  obj.DMP_Params.alpha_z .* ...
                      (obj.DMP_Params.beta_z.*(obj.DMP_Params.goal - trajQuery.y_traj(:, i)) - obj.DMP_Params.tau .* trajQuery.dy_traj(:, i)));
            end
            
            obj.Trajectory = trajQuery;
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
                    w(k, :) = VonMises_Basis(s, obj.Force_Params.Mu(k), obj.Force_Params.Sigma(k)); %use Von-Mises basis
                end
                w = w ./ sum(w,1);
                W = [W, w]; %compress diagonal weight matrix W to an array
                %Output force term data
                g = obj.TrainData{i}.y_train(:, end);
                f_d = (obj.DMP_Params.tau.^2 .* obj.TrainData{i}.ddy_train - ...
                    obj.DMP_Params.alpha_z.*(obj.DMP_Params.beta_z.*(g - obj.TrainData{i}.y_train) - obj.DMP_Params.tau.*obj.TrainData{i}.dy_train)) / ...
                    obj.DMP_Params.r; %cancel amptitude r
                F_d = [F_d, f_d];
            end
            
            %Mapping the input states to polynomial space
            poly_S = zeros(obj.LWR_polyOrder+1, size(S,2));
            for n=0:obj.LWR_polyOrder
                poly_S(n+1, :) = mod(S, 2*pi).^n; %constrained states to 0~2pi
            end
            
            %solution of weights
            obj.Force_Params.weights = zeros(obj.DMP_Params.transVarDim, obj.LWR_polyOrder+1, obj.Force_Params.nbFuncs); %D X n X k 
            for i=1:obj.Force_Params.nbFuncs
                obj.Force_Params.weights(:, :, i) = (F_d.*W(i, :)) * poly_S' / ((poly_S.*W(i, :)) * poly_S');
            end
            
        end
        
        %% init_LWR---------------------------------------------------------------------------
        function init_LWR(obj, rlwr_polyorder, lambda, uncertainty)
            %init_LWR: init recursive locally weighted regression options
            % input: lambda--forgetting factor;  ;
            % uncertainty--uncertainty level of initial P_0 being identity matrix
            obj.LWR_polyOrder = rlwr_polyorder;
            if nargin>1
                obj.lambda = lambda;
                P_dim = obj.LWR_polyOrder + 1;
                
                if ~exist('uncertainty','var')
                    uncertainty = 1000;
                end
                
                obj.P_rlwr = uncertainty * ones(P_dim, P_dim, obj.Force_Params.nbFuncs) .* eye(P_dim); %init P matrix
                obj.Force_Params.weights = zeros(obj.DMP_Params.transVarDim, P_dim, obj.Force_Params.nbFuncs); %D X n X k 
            end
        end
        
          %% RLWR_Train---------------------------------------------------------------------------
        function RLWR_Train(obj)
            %RLWR_Train: train the DMP with recursive locally weighted
            %regression
            obj.TrainMethod = 'RLWR';
            if isempty(obj.LWR_polyOrder)
                obj.LWR_polyOrder = 1;
                disp("LWR_polyOrder empty! Set LWR_polyOrder to default 1.")
            end
            
            if isempty(obj.lambda) || isempty(obj.P_rlwr)
                disp('please init RLWR options first!');
                return
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
                        w(k, :) = VonMises_Basis(s, obj.Force_Params.Mu(k), obj.Force_Params.Sigma(k)); %use Von-Mises basis
                    end
                    w = w ./ sum(w,1);
                    W = [W, w];
                    %Output force term data
                    g = obj.TrainData{id}.y_train(:, end);
                    f_d = (obj.DMP_Params.tau.^2 .* obj.TrainData{id}.ddy_train - ...
                        obj.DMP_Params.alpha_z.*(obj.DMP_Params.beta_z.*(g - obj.TrainData{id}.y_train) - obj.DMP_Params.tau.*obj.TrainData{id}.dy_train))/ ...
                    obj.DMP_Params.r; %cancel amptitude r   
                
                    F_d = [F_d, f_d];
                end
                
                %Mapping the input states to polynomial space
                poly_S = zeros(obj.LWR_polyOrder+1, size(S,2));
                for n=0:obj.LWR_polyOrder
                    poly_S(n+1, :) = mod(S, 2*pi).^n; %constrained states to 0~2pi
                end
                
                %%recursively update the weights
                w_old = obj.Force_Params.weights; %D X n X k matrix
                P_old = obj.P_rlwr;
                for i = 1:size(F_d,2)
                    w = W(:, i);
                    xi = poly_S(:, i);
                    for k=1:obj.Force_Params.nbFuncs
                        w_k = w_old(:, :, k)';
                        obj.P_rlwr(:, :, k) = 1/obj.lambda * (P_old(:, :, k) - (P_old(:, :, k)*(xi * xi' )*P_old(:, :, k))/(obj.lambda / w(k) + xi' * P_old(:, :, k) * xi));
                        error_k = F_d(:, i) - w_k' * xi;
                        obj.Force_Params.weights(:, :, k) = ( w_k + w(k) * (obj.P_rlwr(:, :, k) * xi) * (error_k') )';
                    end
                    w_old = obj.Force_Params.weights;
                    P_old = obj.P_rlwr;
                end
                obj.lastDataId_rlwr = obj.nbDemons;
                
            end
        end
        
        %% RLWR_Reset----------------------------------------------------------------------------
        function RLWR_Reset(obj)
            %RLWR_Reset: reset training result of RLWR
            obj.P_rlwr = [];
            obj.Force_Params.weights = [];
            obj.lastDataId_rlwr = 0;
        end
        
        %% plotResults1D-----------------------------------------------------------------------------
        function plotResults1D(obj, dimId, title, saveFigName)
            %plotResults1D:plot all results of 1D-periodic DMP including demonstrations, predicted
            %   trajectories, force term, activated basis functions ; Init figures and plot parameters
            % Input: dimId--index of 1 dimension;  saveFigName--file name
            
            figure('position',[50,80,1600,900],'color',[1 1 1]);
            
            %y
            subplot(4,1,1);
            hold on; 
            for i=1:size(obj.TrainData, 2)
                t = (0 : obj.TrainData{i}.nbData-1)*obj.dt;
                plot(t, obj.TrainData{i}.y_train(dimId(1),:), '.', 'markersize', 8, 'color', [.7 .7 .7]);
            end
            legend('Demonstrations');
            plot(obj.Trajectory.timeQuery, obj.Trajectory.y_traj(dimId(1),:), '-', 'linewidth', 3, 'color', [.8 0 0], 'DisplayName', 'Prediction');
            legend('FontSize', 12);
            xlabel('t/s', 'FontSize', 16); 
            ylabel(sprintf('$y_%d$', dimId(1)), 'Interpreter', 'latex', 'FontSize', 16);
            
            %dy
            subplot(4,1,2);
            hold on; 
            for i=1:size(obj.TrainData, 2)
                t = (0 : obj.TrainData{i}.nbData-1)*obj.dt;
                plot(t, obj.TrainData{i}.dy_train(dimId(1),:), '.', 'markersize', 8, 'color', [.7 .7 .7]);
            end
            legend('Demonstrations');
            plot(obj.Trajectory.timeQuery, obj.Trajectory.dy_traj(dimId(1),:), '-', 'linewidth', 3, 'color', [.8 0 0], 'DisplayName', 'Prediction');
            legend('FontSize', 12);
            xlabel('t/s', 'FontSize', 16); ylabel(sprintf('$\\dot{y}_%d$', dimId(1)), 'Interpreter', 'latex', 'FontSize', 16);
            
            %ddy
            subplot(4,1,3);
            hold on; 
            for i=1:size(obj.TrainData, 2)
                t = (0 : obj.TrainData{i}.nbData-1)*obj.dt;
                plot(t, obj.TrainData{i}.ddy_train(dimId(1),:), '.', 'markersize', 8, 'color', [.7 .7 .7]);
            end
            legend('Demonstrations');
            plot(obj.Trajectory.timeQuery, obj.Trajectory.ddy_traj(dimId(1),:), '-', 'linewidth', 3, 'color', [.8 0 0], 'DisplayName', 'Prediction');
            legend('FontSize', 12);
            xlabel('t/s', 'FontSize', 16); ylabel(sprintf('$\\ddot{y}_%d$', dimId(1)), 'Interpreter', 'latex', 'FontSize', 16);
            
            %force term
            subplot(4,1,4);
            hold on; 
            plot(obj.Trajectory.timeQuery, obj.Trajectory.f_traj(dimId(1),:), '-', 'linewidth', 3, 'color', [.8 0 0], 'DisplayName', 'Force term');
            legend('FontSize', 12);
            xlabel('t/s', 'FontSize', 16); ylabel(sprintf('$f_%d$', dimId(1)), 'Interpreter', 'latex', 'FontSize', 16);
            
            %add title over subplots
            sgtitle(title);
            
            %Save figures
            if exist('saveFigName', 'var')
                for i = 1:size(saveFigName, 2)
                    saveas(gcf, saveFigName{i});
                end
            end
        end
        
        %% plotResults2D---------------------------------------------------------------------------
        function plotResults2D(obj, dimId, saveFigName)
            %plotResults2D:plot all results of 2D-periodic DMP including demonstrations, predicted
            %   trajectories, force term, activated basis functions ; Init figures and plot parameters
            % Input: dimId--index of 2 dimension;  saveFigName--file name
            
            figure('PaperPosition',[0 0 16 8],'position',[50,80,1600,900],'color',[1 1 1]); 
            xx = round(linspace(1, 64, obj.Force_Params.nbFuncs)); %index to divide colormap
            clrmap = colormap('jet')*0.5;
            clrmap = min(clrmap(xx,:),.9);

            %Plot spatial demonstrations and predicted trajectory
            axes('Position',[0 0 .2 1]); hold on; axis off;
            for i=1:size(obj.TrainData, 2)
            plot(obj.TrainData{i}.y_train(dimId(1),:), obj.TrainData{i}.y_train(dimId(2),:), '.', 'markersize', 8, 'color', [.7 .7 .7]);
            end
            plot(obj.Trajectory.y_traj(dimId(1),:), obj.Trajectory.y_traj(dimId(2),:), '-', 'linewidth', 3, 'color', [.8 0 0]);
            axis equal; axis square; 
            if obj.TrainMethod == 'RLWR'
                title(sprintf("Trained By recursive-LWR\n$\\lambda=%.2f$\nPolynomial order: %d", obj.lambda, obj.LWR_polyOrder), ...
                                'fontsize',16,'interpreter','latex');
            elseif obj.TrainMethod == 'BLWR'
                title(sprintf("Trained By batch-LWR\nPolynomial order: %d", obj.LWR_polyOrder), 'fontsize',16,'interpreter','latex');
            end

            %Timeline plot of the force term
            axes('Position',[.25 .58 .7 .4]); hold on; 
            plot(obj.Trajectory.timeQuery, obj.Trajectory.f_traj(dimId(1),:), '-','linewidth', 2, 'color', [.8 0 0]);
            plot(obj.Trajectory.timeQuery, obj.Trajectory.f_traj(dimId(2),:), '-','linewidth', 2, 'color', [.5 0 0]);
            % axis([min(timeSqe) max(timeSqe) min(trajQuery.f_traj(dimId(1),:)) max(trajQuery.f_traj(dimId(1),:))]);
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
            xlabel('$\phi$','fontsize',16,'interpreter','latex'); 
            ylabel('$\Psi$','fontsize',16,'interpreter','latex');

            %Save figures
            if exist('saveFigName', 'var')
                for i = 1:size(saveFigName, 2)
                    saveas(gcf, saveFigName{i});
                end
            end
        end
        
    end
end

