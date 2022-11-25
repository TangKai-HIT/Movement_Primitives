%periodic/rhythmic DMP test 1: train DMP using RLWR; Demonstration: legs joint
%angles during saggital walking (from mocap data 07)
clc; clear; 
close all;
addpath(genpath('./src'), genpath('./my_util'));
%% Load data
load('./dataset/walkDemons_07_grad.mat');
loadDemonId = 1; %index of chosen demonstrations in the data set

%plot demonstration 1 animation
% figure()
% plot(rad2deg(walkDemons_07{1}.pos(1,:)), rad2deg(walkDemons_07{1}.pos(2,:)));
% xlabel(walkDemons_07{1}.order{1}); ylabel(walkDemons_07{1}.order{2}); 

%% periodic DMP parameters
alpha_z =5;
beta_z = alpha_z/4;
goal = 0;
r = 1;
tau = 132*(1/120)/(pi);
% tau = 1/(0.5*pi); %for harmonic test
dt =1/120;
nbFuncs = 50;
%% Construct training data set & Init DMP model
%construct training set
trainPosData1=cell(1,size(loadDemonId,2));
trainVelData1=cell(1,size(loadDemonId,2));
trainAccelData1=cell(1,size(loadDemonId,2));
for i=1 : size(loadDemonId,2)
    trainPosData1{i} = walkDemons_07{loadDemonId(i)}.pos(1:2, :); %only use left hip & knee
    trainVelData1{i} = walkDemons_07{loadDemonId(i)}.vel(1:2, :);
    trainAccelData1{i} = walkDemons_07{loadDemonId(i)}.accel(1:2, :);
end

%use harmonic function(uncomment)
% t= 0:dt:2;
% test = cos(3*pi*t).*sin(1*pi*t);

%init periodicDMP
pDMP1 = pDMP_ver1(alpha_z, beta_z, tau, goal, r, dt, ...
                                            trainPosData1, trainVelData1, trainAccelData1);
% pDMP1 = periodicDMP_1(alpha_z, beta_z, tau, goal, r, dt, {test}); %for harmonic test
                                        
%init Basis functions and weights                                        
pDMP1.init_Basis_stateBased(nbFuncs); 

%% Train & Predict using recursive-LWR
%train DMP using recursive locally weighted regression with forgetting factor
lambda = 1; %forgetting factor
polyOrder = 1; %set LWR polynomial order
uncertainty = 1000; %uncertainty level of initial P_0 being identity matrix

pDMP1.RLWR_Reset(); %reset training results
pDMP1.init_LWR(polyOrder, lambda, uncertainty);
pDMP1.RLWR_Train();

%%Generate predicted trajectories
endtime = 310*dt;
pDMP1.genPredTraj_LWR(endtime);

%% Plot regression results 
saveFigs_2D = {'results/pDMP_test1/lhipknee_poly1_basis50.fig', 'results/pDMP_test1/lhipknee_poly1_basis50.jpg'};
pDMP1.plotResults2D([1,2], saveFigs_2D); %2-dim
% pDMP1.plotResults2D([1,2]);

saveFigs_1D = {'results/pDMP_test1/lhip_poly1_basis50.fig', 'results/pDMP_test1/lhip_poly1_basis50.jpg'};
pDMP1.plotResults1D(1, 'periodic DMP of left hip', saveFigs_1D); %1-dim
% pDMP1.plotResults1D(1);