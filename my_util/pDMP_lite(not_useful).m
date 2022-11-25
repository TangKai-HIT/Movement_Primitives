 classdef pDMP_lite < handle
    %pDMP_lite: Simulink version of periodic 1-D DMP trained by Recursive LWR  or GMR, with
    %shared 1-dimensional canonical system (a phase oscillator phi)
    %   Describtion-------------------------------------
    %   : 1.lite form of pDMP_ver1, using with simulink;
    %     2.only state-based init function;
  
    
    properties
        DMP_Params=struct('goal', [], 'y_0', [], 'dy_0', [], 'alpha_z', [], 'beta_z', [], ...
                                        'tau', [], 'r', []);
                                    
        Force_Params=struct('nbFuncs', [], 'weights', [], 'Mu', [], 'Sigma', []); 
          
        lambda; %forgetting factor
        
    end
    
    properties (SetAccess = private)
        
        P_rlwr; %P matrix for recursive LWR
        LWR_polyOrder;
        y;
        dy;
        ddy;
    end
    
    methods
        %% pDMP_lite: Constructor-------------------------------------------------------------------
        function obj = pDMP_lite(alpha_z, beta_z, tau, goal, r)
            %pDMP_lite: Construct an instance of this class
            if nargin>0
                %%specify parameters of canonical & transformation systems
                obj.DMP_Params.alpha_z = alpha_z;
                obj.DMP_Params.beta_z = beta_z;
                obj.DMP_Params.tau = tau;
                obj.DMP_Params.goal = goal;
                obj.DMP_Params.r = r;           
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
           
            S = 0: 0.01: 2*pi; 
            TimingSep = linspace(0, 2*pi, nbFuncs+1);
            
            obj.Force_Params.weights = zeros(1, nbFuncs);
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

            if isempty(obj.LWR_polyOrder)
                obj.LWR_polyOrder = 1;
                disp("LWR_polyOrder empty! Set LWR_polyOrder to default 1.")
            end
            
            if isempty(obj.lambda) || isempty(obj.P_rlwr)
                disp('please init RLWR options first!');
                return
            end
       
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
                    f_d = obj.DMP_Params.tau.^2 .* obj.TrainData{id}.ddy_train - ...
                        obj.DMP_Params.alpha_z.*(obj.DMP_Params.beta_z.*(g - obj.TrainData{id}.y_train) - obj.DMP_Params.tau.*obj.TrainData{id}.dy_train)/ ...
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

        end
        
        %% RLWR_Reset----------------------------------------------------------------------------
        function RLWR_Reset(obj)
            %RLWR_Reset: reset training result of RLWR
            obj.P_rlwr = [];
            obj.Force_Params.weights = [];
        end
               
    end
end

