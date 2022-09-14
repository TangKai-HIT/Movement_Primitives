classdef DMP_Base1 < handle
    %DMP_Base1: type 1 Base class of classic DMP
    %   Describtion-------------------------------------
    %   
    
    properties
        DMP_Params=struct('goal', [], 'y_0', [], 'dy_0', [], 'alpha_z', 25, 'beta_z', 25/4, ...
                                        'alpha_x', 25/3, 'tau', [], 'canonVarDim', 1, 'transVarDim', 1);
                                    
        Force_Params=struct('nbFuncs', 5, 'weights', [], 'Mu', [], 'Sigma', []);
        
        Trajectory;
        
        TrainData = {};
        
        nbDemons; %number of demonstrations
        dt;
    end
    
    properties (Access = private)
        TrainDataTemplate = struct('y_train', [], 'dy_train', [], 'ddy_train', [], 'nbData', []); %template of each demonstration data
        TrajectoryTemplate = struct('s_traj', [], 'y_traj', [], 'dy_traj', [], 'ddy_traj', [], 'f_traj', [], 'activations', []); %template of query trajectory data
        P_rlwr; %P matrix for recursive LWR
        P_rls; %P matrix for recursive least square
    end
    
    methods
        function obj = DMP_Base1(alpha_z, beta_z, tau, alpha_x, dt, ...
                                            trainPosData, trainVelData, trainAccelData)
            %DMP_Base1 Construct an instance of this class
            %   inputs: trainPosData--cell array containing demonstrations
            %   position data Yi (Yi: d X N matrix)
            if nargin>0
                %%specify parameters of canonical & transformation systems
                obj.DMP_Params.alpha_z = alpha_z;
                obj.DMP_Params.beta_z = beta_z;
                obj.DMP_Params.tau = tau;
                obj.DMP_Params.alpha_x = alpha_x;
%                 obj.DMP_Params.canonVarDim = canonVarDim; %dimension of canonical variable x (or s)
                obj.DMP_Params.transVarDim = size(trainPosData{1}, 1); %dimension of ouput variable y
                
                %%specify time increment
                obj.dt = dt;
                
                if nargin>5 % if input training data
                    %%specify training data(in batch)
                    obj.nbDemons = size(trainPosData,2);
                    
                    for i=1:obj.nbDemons
                        obj.TrainData{i} = obj.TrainDataTemplate; %assign empty template
                        %obj.TrainData{i}.y_train = spline(1:size(trainPosData{i},2), trainPosData{i}, linspace(1,size(trainPosData{i},2),nbSampData)); %Resampling
                        obj.TrainData{i}.y_train = trainPosData{i};
                        obj.TrainData{i}.nbData = size(trainPosData{i}, 2);
                        if ~exist('trainVelData','var')
                            obj.TrainData{i}.dy_train = gradient(trainPosData{i})/dt;
                        else
                            obj.TrainData{i}.dy_train = trainVelData{i};
                        end
                        if ~exist('trainAccelData','var')
                            obj.TrainData{i}.ddy_train = gradient(obj.TrainData{i}.dy_train)/dt;
                        else
                            obj.TrainData{i}.ddy_train = trainAccelData{i};
                        end
                    end
                    
                    %set default goal & initial states
                    obj.DMP_Params.goal = obj.TrainData{1}.y_train(:, end); %set default goal
                    obj.DMP_Params.y_0 = obj.TrainData{1}.y_train(:, 1); %set default initial state
                    obj.DMP_Params.dy_0 = obj.TrainData{1}.dy_train(:, 1); %set default initial state
                end
                
            end
        end
        
        function init_RBFBasis_timeBased(obj, nbFuncs)
            %init_RBFBasis_timeBased: init RBF basis functions to evenly distributed on time interval.
            %   Inputs: nbFuncs--number of basis functions
            
%             params_diagRegFact = 1E-4; %Optional regularization term to avoid numerical instability
            obj.Force_Params.Mu = zeros(1,5);
            obj.Force_Params.Sigma = zeros(1,5);
            
            for i=1:nbFuncs
                obj.Force_Params.Mu(i) = exp(-obj.DMP_Params.alpha_x / obj.DMP_Params.tau *(i-1)/(nbFuncs-1));
                if i > 1
                    obj.Force_Params.Sigma(i-1) = ((obj.Force_Params.Mu(i) - obj.Force_Params.Mu(i-1))/2)^2;
                end
            end
            obj.Force_Params.Sigma(nbFuncs) = obj.Force_Params.Sigma(nbFuncs-1);
            
            obj.Force_Params.weights = ones(obj.DMP_Params.transVarDim, nbFuncs) * (1/nbFuncs);
        end
        
        function s_Query=genCanonStates(obj, timeQuery)
            %genCanonStates: generate canonical system states by query timing points(i.e. trajectory of x)
            %   inputs:
            s_Query = exp(-obj.DMP_Params.alpha_x/obj.DMP_Params.tau*timeQuery); 
        end
        
        function [trajQuery, timeQuery]=genPredTraj(obj, endTime)
            %genPredTraj: generate predicted output states by query timing points(i.e. trajectory of y)
            %   inputs:
            
            trajQuery  = obj.TrajectoryTemplate; %init using trajectory template structure
            %generate canonical states
            timeQuery = 0: obj.dt: endTime;
            trajQuery.s_traj = obj.genCanonStates(timeQuery); 
            %compute Phi matrix(basis functions)
            Phi = zeros(obj.Force_Params.nbFuncs, size(timeQuery,2));
            for k =1:obj.Force_Params.nbFuncs
                Phi(k, :) = rbf_Basis(trajQuery.s_traj, obj.Force_Params.Mu(k), obj.Force_Params.Sigma(k));
            end
            trajQuery.activations = Phi; %record activations
            Phi = Phi ./ sum(Phi,1) .* trajQuery.s_traj;
            %compute force terms
            trajQuery.f_traj = obj.Force_Params.weights * Phi;
            
            %Reproduction: Eular forward iteration
            N = size(timeQuery,2); %number of query points
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
        
        function LS_batchTrain(obj)
            %LS_batchTrain: train the DMP with batch least square 
            
            %%construct input data & output data matrices for normal equation
            X_in=[];
            Y_out = [];
            for i=1:obj.nbDemons
                %canonical states for current demonstration
                s = obj.genCanonStates([0 : obj.TrainData{i}.nbData-1]*obj.dt);
                %Input demonstrated data
                Phi = zeros(obj.Force_Params.nbFuncs, obj.TrainData{i}.nbData);
                for k =1:obj.Force_Params.nbFuncs
                    Phi(k, :) = rbf_Basis(s, obj.Force_Params.Mu(k), obj.Force_Params.Sigma(k));
                end
                Phi = Phi ./ sum(Phi,1) .* s;
                X_in = [X_in, Phi];
                %Output force term data
                g = obj.TrainData{i}.y_train(:, end);
                f_d = obj.DMP_Params.tau^2 * obj.TrainData{i}.ddy_train - ...
                    obj.DMP_Params.alpha_z*(obj.DMP_Params.beta_z*(g - obj.TrainData{i}.y_train) - obj.DMP_Params.tau*obj.TrainData{i}.dy_train);
                
                Y_out = [Y_out, f_d];
            end
            %take transpose the matrices to get standard design matrices
            X_in = X_in';
            Y_out = Y_out';
            %minimum norm solution of the normal equation
            obj.Force_Params.weights = (pinv(X_in)*Y_out)';
        end
        
        function LWR_batchTrain(obj)
            %LWR_batchTrain: train the DMP with batch locally weighted
            %regression
            
        end
        
          function RLS_Train(obj)
            %LWR_batchTrain: train the DMP with recursive locally weighted
            %regression
            
          end
        
        function RLWR_Train(obj)
        %LWR_batchTrain: train the DMP with recursive locally weighted
        %regression

        end
        
    end
end
