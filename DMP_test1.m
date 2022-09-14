% DMP test demo1: train DMP using batch least square
addpath(genpath('./src'), genpath('./my_util'));
%% load data
load('./dataset/G.mat');
loadDemonId = 1:4; %index of chosen demonstrations in the data set
%% DMP parameters
alpha_z =25;
beta_z = alpha_z/4;
alpha_x = alpha_z/3;
tau = alpha_x;
dt =0.01;
nbFuncs = 5;
%% Trained by least square (batch)
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
DMP1 = DMP_Base1(alpha_z, beta_z, tau, alpha_x, dt, ...
                                            trainPosData1, trainVelData1, trainAccelData1);
%init Basis functions and weights                                        
DMP1.init_RBFBasis_timeBased(nbFuncs);                                       
%train DMP using  batch least square
DMP1.LS_batchTrain();
%% generate predicted trajectories
endtime = 200*dt;
[predTraj, timeSqe]= DMP1.genPredTraj(endtime);