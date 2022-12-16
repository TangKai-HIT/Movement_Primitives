%TP-GMM test1 with 2 frames
clc; clear; close;
addpath(genpath("../../src/"))

%% Parameters
numGauss = 3; %Number of Gaussians in the GMM
numDemons = 4; %Number of demonstrations
numFrames = 2;
dim = 2;
myTPGMM = TPGMM(numGauss, numFrames, dim);

%% Get frame data
load('data/Data01.mat');

%Get raw data & Frame pose data
numData = size(s(1).Data0,2);
rawData = cell(1, numDemons); 
for n=1:numDemons
	rawData{n} = s(n).Data0(2:end,:);%Remove time
    for m =1:numFrames
        framePose(n).A{m} = s(n).p(m).A;
        framePose(n).b{m} = s(n).p(m).b;
    end
end

%Get frame data
frameData = myTPGMM.getFrameData(rawData', framePose);

%% TP-GMM learning
myTPGMM.initTPGMM_kmeans(frameData);

tic;
[likelihoodHistory, iters, exitFlag] = myTPGMM.train_EM(frameData); %perform EM
solveTime = toc;
fprintf("Run time: %.3f\n", solveTime);

%% Reconstruct GMM for each demonstration
for n=1:numDemons
	[s(n).Mu, s(n).Sigma] = myTPGMM.retrieveTPGMM(s(n).p);
end

%% Plots
figure('position',[10,10,2300,900]);
xx = round(linspace(1,64,numDemons));
clrmap = colormap('jet');
clrmap = min(clrmap(xx,:),.95);
limAxes = [-1.2 0.8 -1.1 0.9];
colPegs = [0.2863 0.0392 0.2392; 0.9137 0.4980 0.0078];

%DEMOS
subplot(1,numFrames+1,1); hold on; box on; title('Demonstrations');
for n=1:numDemons
	%Plot frames
	for m=1:numFrames
		plotPegs(s(n).p(m), colPegs(m,:));
	end
	%Plot trajectories
	plot(s(n).Data(2,1), s(n).Data(3,1),'.','markersize',15,'color',clrmap(n,:));
	plot(s(n).Data(2,:), s(n).Data(3,:),'-','linewidth',1.5,'color',clrmap(n,:));
	%Plot Gaussians
	plotGMM2D(gca, s(n).Mu, s(n).Sigma, [.5 .5 .5],.8);
end
axis(limAxes); axis square; set(gca,'xtick',[],'ytick',[]);

%% Function to plot pegs ----------------------------------------------------------------------------------------------------------------------
function h = plotPegs(p, colPegs, fa)
	if ~exist('colPegs','var')
		colPegs = [0.2863 0.0392 0.2392; 0.9137 0.4980 0.0078];
	end
	if ~exist('fa', 'var')
		fa = .6;
	end
	pegMesh = [-4 -3.5; -4 10; -1.5 10; -1.5 -1; 1.5 -1; 1.5 10; 4 10; 4 -3.5; -4 -3.5]' *1E-1;
	for m=1:length(p)
		dispMesh = p(m).A * pegMesh + repmat(p(m).b,1,size(pegMesh,2));
		h(m) = patch(dispMesh(1,:),dispMesh(2,:),colPegs(m,:),'linewidth',1,'edgecolor','none','facealpha',fa);
	end
end