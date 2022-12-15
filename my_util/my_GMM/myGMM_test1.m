%Test GMM with hand-written 'G' letters
clc; clear; close all;

%% Parameters
numGauss = 5; %Number of Gaussians
dim = 2; %Number of variables [x1,x2]
numData = 200; %Length of each trajectory
numSamples = 5; %Number of demonstrations

%% Load handwriting data
demos=[];
load('../../dataset/G.mat');
%nbSamples = length(demos);
Data=[];
for n=1:numSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),numData)); %Resampling
	%s(n).Data = interp1(1:size(demos{n}.pos,2), demos{n}.pos', linspace(1,size(demos{n}.pos,2),nbData));
	Data = [Data s(n).Data]; 
end
Data = Data';

%% Parameters estimation using GMM
gmm = myGMM(numGauss);
gmm.initGMM_kmeans(Data); %init GMM
tic;
[likelihoodHistory, iters, exitFlag] = gmm.train_EM(Data); %perform EM
solveTime = toc;
fprintf("Run time: %.3f\n", solveTime);

%% Plots
figure('position',[10,10,700,500]); hold on; axis off;
plot(Data(:,1),Data(:,2),'.','markersize',8,'color',[.5 .5 .5]);
gmm.plotGMM2D(gca, [.8 0 0], .5);
axis equal; set(gca,'Xtick',[]); set(gca,'Ytick',[]);