function demo_DMP_GMR04
% Enhanced dynamic movement primitive (DMP) model trained with EM by using a Gaussian mixture 
% model (GMM) representation, with full covariance matrices coordinating the different variables 
% in the feature space, and by using the task-parameterized model formalism. After learning 
% (i.e., autonomous organization of the basis functions (position and spread), Gaussian mixture 
% regression (GMR) is used to regenerate the path of a spring-damper system, resulting in a 
% nonlinear force profile. 
%
% If this code is useful for your research, please cite the related publication:
% @incollection{Calinon19chapter,
% 	author="Calinon, S. and Lee, D.",
% 	title="Learning Control",
% 	booktitle="Humanoid Robotics: a Reference",
% 	publisher="Springer",
% 	editor="Vadakkepat, P. and Goswami, A.", 
% 	year="2019",
% 	doi="10.1007/978-94-007-7194-9_68-1",
% 	pages="1--52"
% }
% 
% Copyright (c) 2019 Idiap Research Institute, http://idiap.ch/
% Written by Sylvain Calinon, http://calinon.ch/
% 
% This file is part of PbDlib, http://www.idiap.ch/software/pbdlib/
% 
% PbDlib is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License version 3 as
% published by the Free Software Foundation.
% 
% PbDlib is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with PbDlib. If not, see <http://www.gnu.org/licenses/>.

addpath('./m_fcts/');


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.nbStates = 5; %Number of states in the GMM
model.nbVar = 3; %Number of variables [s,F1,F2] (decay term and perturbing force)
model.nbFrames = 1; %Number of candidate frames of reference (centered on goal position)
model.kP = 50; %Stiffness gain
model.kV = (2*model.kP)^.5; %Damping gain (with ideal underdamped damping ratio)
model.alpha = 1.0; %Decay factor
model.dt = 0.01; %Duration of time step
model.nbVarPos = model.nbVar-1; %Dimension of spatial variables
nbData = 200; %Length of each trajectory
nbSamples = 4; %Number of demonstrations
L = [eye(model.nbVarPos)*model.kP, eye(model.nbVarPos)*model.kV]; %Feedback term
%Create transformation matrix to compute r(1).currTar = x + dx*kV/kP + ddx/kP
K1d = [1, model.kV/model.kP, 1/model.kP];
K = kron(K1d,eye(model.nbVarPos));


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/2Dletters/G.mat');
sIn(1) = 1; %Initialization of decay term
for t=2:nbData
	sIn(t) = sIn(t-1) - model.alpha * sIn(t-1) * model.dt; %Update of decay term (ds/dt=-alpha s)
end
DataDMP = zeros(model.nbVar,1,nbData*nbSamples);
Data=[];
for n=1:nbSamples
	%Task parameters (canonical coordinate system centered on the end-trajectory target)
	s(n).p(1).A = eye(model.nbVar);
	s(n).p(1).b = [0; demos{n}.pos(:,end)];
	%Demonstration data as [x;dx;ddx]
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	Data = [Data s(n).Data]; %Original data
	s(n).Data = [s(n).Data; gradient(s(n).Data)/model.dt]; %Velocity computation
	s(n).Data = [s(n).Data; gradient(s(n).Data(end-model.nbVarPos+1:end,:))/model.dt]; %Acceleration computation
	s(n).Data = [sIn; K*s(n).Data]; %r(1).currTar computation
	s(n).Data = s(n).p(1).A \ (s(n).Data - repmat(s(n).p(1).b,1,nbData)); %Observation from the perspective of the frame
	DataDMP(:,1,(n-1)*nbData+1:n*nbData) = s(n).Data; %Training data as [s;r(1).currTar]
end


%% Learning and reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%model = init_tensorGMM_kmeans(DataDMP, model); 
model = init_tensorGMM_timeBased(DataDMP, model);  
model = EM_tensorGMM(DataDMP, model);

%Task-adaptive spring-damper attractor path retrieval
r(n).p(1).A = eye(model.nbVar);
r(n).p(1).b = s(n).p(1).b + [0; 0; 5]; %Offset added to test generalization capability
[r(n).Mu, r(n).Sigma] = productTPGMM0(model, r(n).p); 
r(n).Priors = model.Priors;
r(n).nbStates = model.nbStates;
r(1).currTar = GMR(r(n), sIn, 1, 2:model.nbVar);
	
%Motion retrieval with DMP
x = Data(1:model.nbVarPos,1);
dx = zeros(model.nbVarPos,1);
for t=1:nbData
	%Compute acceleration, velocity and position	
	ddx =  L * [r(1).currTar(:,t)-x; -dx]; %Spring-damper system
	dx = dx + ddx * model.dt;
	x = x + dx * model.dt;
	r(1).Data(:,t) = x;
end
	

%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,1300,450],'color',[1 1 1]); 
xx = round(linspace(1,64,model.nbStates));
clrmap = colormap('jet')*0.5;
clrmap = min(clrmap(xx,:),.9);

%Activation of the basis functions
for i=1:model.nbStates
	h(i,:) = model.Priors(i) * gaussPDF(sIn, model.Mu(1,i), model.Sigma(1,1,i));
end
h = h ./ repmat(sum(h,1)+realmin, model.nbStates, 1);

%Spatial plot
subplot(2,4,[1,5]); hold on; axis off;
plot(Data(1,:),Data(2,:),'.','markersize',8,'color',[.7 .7 .7]);
plot(r(1).currTar(1,:), r(1).currTar(2,:), '-','linewidth',2,'color',[1 .7 .7]); %Attractor path
plot(r(1).Data(1,:),r(1).Data(2,:),'-','linewidth',3,'color',[.8 0 0]); %Retrieved path
plot(r(n).p(1).b(2),r(n).p(1).b(3),'k+','linewidth',2,'markersize',12);
axis equal; 

%Timeline plot of the nonlinear perturbing force
subplot(2,4,[2:4]); hold on;
for n=1:nbSamples
	plot(sIn, DataDMP(2,(n-1)*nbData+1:n*nbData), '-','linewidth',2,'color',[.7 .7 .7]);
end
for i=1:model.nbStates
	plotGMM(model.Mu(1:2,i), model.Sigma(1:2,1:2,i), clrmap(i,:), .7);
end
plot(sIn, r(1).currTar(1,:), '-','linewidth',2,'color',[.8 0 0]);
axis([0 1 min(DataDMP(2,:)) max(DataDMP(2,:))]);
ylabel('$\hat{x}_1$','fontsize',16,'interpreter','latex');
view(180,-90);

%Timeline plot of the basis functions activation
subplot(2,4,[6:8]); hold on;
for i=1:model.nbStates
	patch([sIn(1), sIn, sIn(end)], [0, h(i,:), 0], min(clrmap(i,:)+0.5,1), 'EdgeColor', 'none', 'facealpha', .4);
	plot(sIn, h(i,:), 'linewidth', 2, 'color', min(clrmap(i,:)+0.2,1));
end
axis([0 1 0 1]);
xlabel('$s$','fontsize',16,'interpreter','latex'); 
ylabel('$h$','fontsize',16,'interpreter','latex');
view(180,-90);

%print('-dpng','graphs/demo_DMP_GMR04.png');
pause;
close all;