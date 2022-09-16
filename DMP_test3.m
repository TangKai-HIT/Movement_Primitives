% DMP test demo3: train DMP using recursive least square with forgetting factor
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
%% Trained by recursive least square
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
% DMP1.init_RBFBasis_stateBased(nbFuncs); 

%train DMP using recursive least square with forgetting factor
DMP1.RLS_Train(0.99);

%% Generate predicted trajectories
endtime = 200*dt;
[predTraj, timeSqe]= DMP1.genPredTraj(endtime);

%% Plot local linear regression results 
%Init figures and plot parameters
figure('PaperPosition',[0 0 16 8],'position',[50,80,1600,900],'color',[1 1 1]); 
xx = round(linspace(1, 64, DMP1.Force_Params.nbFuncs)); %index to divide colormap
clrmap = colormap('jet')*0.5;
clrmap = min(clrmap(xx,:),.9);

%Plot spatial demonstrations and predicted trajectory
axes('Position',[0 0 .2 1]); hold on; axis off;
for i=1:size(DMP1.TrainData, 2)
plot(DMP1.TrainData{i}.y_train(1,:), DMP1.TrainData{i}.y_train(2,:), '.', 'markersize', 8, 'color', [.7 .7 .7]);
end
plot(predTraj.y_traj(1,:), predTraj.y_traj(2,:), '-', 'linewidth', 3, 'color', [.8 0 0]);
axis equal; axis square;  

%Timeline plot of the force term
axes('Position',[.25 .58 .7 .4]); hold on; 
plot(timeSqe, predTraj.f_traj(1,:), '-','linewidth', 2, 'color', [.8 0 0]);
plot(timeSqe, predTraj.f_traj(2,:), '-','linewidth', 2, 'color', [.5 0 0]);
% axis([min(timeSqe) max(timeSqe) min(trajQuery.f_traj(1,:)) max(trajQuery.f_traj(1,:))]);
legend('$f_1(x)$','$f_2(x)$','fontsize',12,'interpreter','latex')
ylabel('$Force$','fontsize',16,'interpreter','latex');
xlabel('$t/s$','fontsize',16,'interpreter','latex');
view(180,-90);

%Plot of the basis functions activation w.r.t canonical state
axes('Position',[.25 .12 .7 .4]); hold on; 
for i=1:DMP1.Force_Params.nbFuncs
	patch([predTraj.s_traj(1), predTraj.s_traj, predTraj.s_traj(end)], ...
                [0, predTraj.activations(i,:), 0], min(clrmap(i,:)+0.5,1), 'EdgeColor', 'none', 'facealpha', .4);
	plot(predTraj.s_traj, predTraj.activations(i,:), 'linewidth', 2, 'color', min(clrmap(i,:)+0.2,1));
end
% axis([min(sIn) max(sIn) 0 1]);
xlabel('$x$','fontsize',16,'interpreter','latex'); 
ylabel('$\Psi$','fontsize',16,'interpreter','latex');
view(180,-90);

%Save figures
saveas(gcf, 'results/test3/test3_RLS_2.fig');
saveas(gcf, 'results/test3/test3_RLS_2.jpg');