% DMP test demo5: train DMP using batch/recursive locally weighted regression(polynomial) with forgetting factor
clc; clear; close all;
addpath(genpath('./src'), genpath('./my_util'));
%% Load data
load('./dataset/G.mat');
loadDemonId = 1:4; %index of chosen demonstrations in the data set
%% DMP parameters
alpha_z =10;
beta_z = alpha_z/4;
alpha_x = 1;
tau = 0.5;
dt =0.01;
nbFuncs = 5;
%% Construct training data set & Init DMP model
%construct training set
trainPosData1=cell(1,size(loadDemonId,2));
trainVelData1=cell(1,size(loadDemonId,2));
trainAccelData1=cell(1,size(loadDemonId,2));
for i=1 : size(loadDemonId,2)
    trainPosData1{i} = demos{loadDemonId(i)}.pos;
    trainVelData1{i} = demos{loadDemonId(i)}.vel;
    trainAccelData1{i} = demos{loadDemonId(i)}.acc;
end

%init DMP
DMP1 = DMP_ver1(alpha_z, beta_z, tau, alpha_x, dt, ...
                                            trainPosData1, trainVelData1, trainAccelData1);
                                        
%init Basis functions and weights                                        
DMP1.init_RBFBasis_timeBased(nbFuncs); 
% DMP1.init_RBFBasis_stateBased(nbFuncs); 

%% Train & Predict using recursive-LWR
%train DMP using recursive locally weighted regression with forgetting factor
lambda = 1; %forgetting factor
DMP1.LWR_polyOrder = 2; %set LWR polynomial order
DMP1.RLWR_Train(lambda);

%%Generate predicted trajectories
endtime = 200*dt;
DMP1.genPredTraj_LWR(endtime);

%%Plot regression results 
saveFigs = {'results/test5/test5_RLWR_poly2_1.fig', 'results/test5/test5_RLWR_poly2_1.jpg'};
% saveFigs = {'results/test5/test5_RLWR_poly2_2.fig', 'results/test5/test5_RLWR_poly2_2.jpg'};
DMP1.plot_Results(saveFigs);

%% Train & Predict using batch-LWR
%train DMP using recursive locally weighted regression with forgetting factor
DMP1.LWR_polyOrder = 2; %set LWR polynomial order
DMP1.LWR_batchTrain();

%%Generate predicted trajectories
endtime = 200*dt;
DMP1.genPredTraj_LWR(endtime);

%%Plot regression results 
saveFigs = {'results/test5/test5_BLWR_poly2_1.fig', 'results/test5/test5_BLWR_poly2_1.jpg'};
% saveFigs = {'results/test5/test5_BLWR_poly2_1.fig', 'results/test5/test5_BLWR_poly2_1.jpg'};
DMP1.plot_Results(saveFigs);