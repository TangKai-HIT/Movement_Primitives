function pDMP_trans1(block)
% level2 s-function of 1-D periodic DMP learning by RLWR, using Von-Mises basis
% input: phi, [y_r; dy_r; ddy_r]; 
% output: y, dy, ddy
% parameters: tau, alpha_z, beta_z, r, y_0, dy_0, nbFunc, lambda, polyOrder, uncertainty
%   Author: Tang kai

%%
%% The setup method is used to set up the basic attributes of the
%% S-function such as ports, parameters, etc. Do not add any other
%% calls to the main body of the function.
%%
setup(block);

%endfunction

%% Function: setup ===================================================
%% Abstract:
%%   Set up the basic characteristics of the S-function block such as:
%%   - Input ports
%%   - Output ports
%%   - Dialog parameters
%%   - Options
%%
%%   Required         : Yes
%%   C MEX counterpart: mdlInitializeSizes
%%
function setup(block)

% Register number of ports
block.NumInputPorts  = 2;
block.NumOutputPorts = 2;

% Setup port properties to be inherited or dynamic
block.SetPreCompInpPortInfoToDynamic;
block.SetPreCompOutPortInfoToDynamic;

% Override input port properties
%input1: phi (canonical system output)
block.InputPort(1).Dimensions        = 1;
block.InputPort(1).DatatypeID  = 0;  % double
block.InputPort(1).Complexity  = 'Real';
block.InputPort(1).DirectFeedthrough = true;
%input2: [y_r; dy_r; ddy_r] (reference input)
block.InputPort(2).Dimensions        = 3;
block.InputPort(2).DatatypeID  = 0;  % double
block.InputPort(2).Complexity  = 'Real';
block.InputPort(2).DirectFeedthrough = true;

% Override output port properties
%y
block.OutputPort(1).Dimensions       = 1;
block.OutputPort(1).DatatypeID  = 0; % double
block.OutputPort(1).Complexity  = 'Real';
%dy
block.OutputPort(2).Dimensions       = 1;
block.OutputPort(2).DatatypeID  = 0; % double
block.OutputPort(2).Complexity  = 'Real';

block.OutputPort(1).SamplingMode = 'Sample';
block.OutputPort(2).SamplingMode = 'Sample';

% Register parameters
block.NumDialogPrms     = 10; %tau, alpha_z, beta_z, r, y_0, dy_0, nbFunc, lambda, polyOrder, uncertainty

% Register sample times
%  [0 offset]            : Continuous sample time
%  [positive_num offset] : Discrete sample time
%
%  [-1, 0]               : Inherited sample time
%  [-2, 0]               : Variable sample time
block.SampleTimes = [-1 0]; 

% Specify the block simStateCompliance. The allowed values are:
%    'UnknownSimState', < The default setting; warn and assume DefaultSimState
%    'DefaultSimState', < Same sim state as a built-in block
%    'HasNoSimState',   < No sim state
%    'CustomSimState',  < Has GetSimState and SetSimState methods
%    'DisallowSimState' < Error out when saving or restoring the model sim state
block.SimStateCompliance = 'DefaultSimState';

%% -----------------------------------------------------------------
%% The MATLAB S-function uses an internal registry for all
%% block methods. You should register all relevant methods
%% (optional and required) as illustrated below. You may choose
%% any suitable name for the methods and implement these methods
%% as local functions within the same file. See comments
%% provided for each function for more information.
%% -----------------------------------------------------------------

block.RegBlockMethod('PostPropagationSetup',    @DoPostPropSetup);
block.RegBlockMethod('InitializeConditions', @InitializeConditions);
block.RegBlockMethod('Derivatives', @Derivatives);  %update derivatives of continuous states
% block.RegBlockMethod('Start', @Start);
block.RegBlockMethod('Outputs', @Outputs);     % Required
block.RegBlockMethod('Update', @Update);
block.RegBlockMethod('Terminate', @Terminate); % Required
block.RegBlockMethod('SetInputPortSamplingMode',@SetInputPortSamplingMode);
%end setup

%%
%% PostPropagationSetup:
%%   Functionality    : Setup work areas and state variables. Can
%%                      also register run-time methods here
%%   Required         : No
%%   C MEX counterpart: mdlSetWorkWidths
%%
function DoPostPropSetup(block)
%work variables: weights for each basis functions
% parameters: tau, alpha_z, beta_z, r, y_0, dy_0, nbFunc, lambda, polyOrder, uncertainty
nbFunc = block.DialogPrm(7).Data; 
nbCoef = block.DialogPrm(9).Data + 1;

%%Discrete work variables
block.NumDworks = nbFunc+2;
for i=1:nbFunc
  block.Dwork(i).Name            = sprintf('basis%d_weights', i);
  block.Dwork(i).Dimensions      = nbCoef;
  block.Dwork(i).DatatypeID      = 0;      % double
  block.Dwork(i).Complexity      = 'Real'; % real
  block.Dwork(i).UsedAsDiscState = true;
end

block.Dwork(nbFunc+1).Name            = 'Mu';
block.Dwork(nbFunc+1).Dimensions      = nbFunc;
block.Dwork(nbFunc+1).DatatypeID      = 0;      % double
block.Dwork(nbFunc+1).Complexity      = 'Real'; % real
block.Dwork(nbFunc+1).UsedAsDiscState = true;

block.Dwork(nbFunc+2).Name            = 'Sigma';
block.Dwork(nbFunc+2).Dimensions      = nbFunc;
block.Dwork(nbFunc+2).DatatypeID      = 0;      % double
block.Dwork(nbFunc+2).Complexity      = 'Real'; % real
block.Dwork(nbFunc+2).UsedAsDiscState = true;

% block.Dwork(nbFunc+3).Name            = 'Force';
% block.Dwork(nbFunc+3).Dimensions      = 1;
% block.Dwork(nbFunc+3).DatatypeID      = 0;      % double
% block.Dwork(nbFunc+3).Complexity      = 'Real'; % real
% block.Dwork(nbFunc+3).UsedAsDiscState = true;

%%Continuous states
block.NumContStates = 2;

%%
%% InitializeConditions:
%%   Functionality    : Called at the start of simulation and if it is 
%%                      present in an enabled subsystem configured to reset 
%%                      states, it will be called when the enabled subsystem
%%                      restarts execution to reset the states.
%%   Required         : No
%%   C MEX counterpart: mdlInitializeConditions
%%
function InitializeConditions(block)
%init continuous states; init weights and basis functions parameters
% parameters: tau, alpha_z, beta_z, r, y_0, dy_0, nbFunc, lambda, polyOrder
%%Continuous states
block.ContStates.Data(1) = block.DialogPrm(5).Data; %y_0
block.ContStates.Data(2) = block.DialogPrm(6).Data; %dy_0

%%Discrete states
nbFunc = block.DialogPrm(7).Data; 
nbCoef = block.DialogPrm(9).Data + 1;
for i=1:nbFunc
    block.Dwork(i).Data = zeros(nbCoef, 1);
end
block.Dwork(nbFunc+1).Data = zeros(nbFunc, 1);
block.Dwork(nbFunc+2).Data = zeros(nbFunc, 1);

%initialize basis functions parameters
TimingSep = linspace(0, 2*pi, nbFunc+1);
S = 0: 0.01: 2*pi;
for i=1:nbFunc
    idtmp = find( S>=TimingSep(i) & S<TimingSep(i+1));
    block.Dwork(i).Data(1) = 1/nbFunc;
    block.Dwork(nbFunc+1).Data(i) = mean(S(idtmp)); %Mu
    block.Dwork(nbFunc+2).Data(i) = cov(S(idtmp)) + 1e-5; %Sigma
end

% block.Dwork(nbFunc+3).Data = 0; % Force
%end InitializeConditions

%% SetInputPortSamplingMode callback
function SetInputPortSamplingMode(block, idx, fd)
block.InputPort(idx).SamplingMode = fd;
block.InputPort(idx).SamplingMode = fd;

%%
%% Outputs:
%%   Functionality    : Called to generate block outputs in
%%                      simulation step
%%   Required         : Yes
%%   C MEX counterpart: mdlOutputs
%%
function Outputs(block)
%Outputs: continuous states y, dy
% parameters: tau, alpha_z, beta_z, r, y_0, dy_0, nbFunc, lambda, polyOrder

block.OutputPort(1).Data = block.ContStates.Data(1); % y 
block.OutputPort(2).Data = block.ContStates.Data(2); % dy 
%end Outputs

%% Derivatives:
%%   Functionality    : update derivatives of continuous states
%%                                  (when continuous states are used)
%%   Required         : No
function Derivatives(block)
tau = block.DialogPrm(1).Data;
alpha_z = block.DialogPrm(2).Data;
beta_z = block.DialogPrm(3).Data;
nbFuncs = block.DialogPrm(7).Data; 
nbCoef = block.DialogPrm(9).Data + 1; %polyOder + 1

y = block.ContStates.Data(1);
dy = block.ContStates.Data(2);

%%compute force term
phi = block.InputPort(1).Data;
%weights
W = VonMises_Basis(phi, block.Dwork(nbFuncs+1).Data, block.Dwork(nbFuncs+2).Data); %use Von-Mises basis
W = W ./ sum(W,1);
%input polynomial
n=0:nbCoef-1;
poly_S = mod(phi, 2*pi).^(n'); %constrained states to 0~2pi
%force
f=0;
for k=1:nbFuncs
    f = f + W(k) * (block.Dwork(nbFuncs).Data' *  poly_S);
end
f = f * block.DialogPrm(4).Data; %r

% update dy
block.Derivatives.Data(1) = dy;  
% update ddy
block.Derivatives.Data(2) = 1/(tau^2) * (f + alpha_z * (beta_z.*(0 - y) - tau * dy));
%end Derivatives

%%
%% Update:
%%   Functionality    : Called to update discrete states
%%                      during simulation step
%%   Required         : No
%%   C MEX counterpart: mdlUpdate
%%
function Update(block)
%RLWR learning
% parameters: tau, alpha_z, beta_z, r, y_0, dy_0, nbFunc, lambda, polyOrder, uncertainty
persistent P_rlwr; 

nbCoef = block.DialogPrm(9).Data + 1; %polyOder + 1
uncertainty = block.DialogPrm(10).Data; %uncertainty of initial P_rlwr being identity matrix
nbFuncs = block.DialogPrm(7).Data;

if isempty(P_rlwr)
        P_rlwr = uncertainty * ones(nbCoef, nbCoef, nbFuncs) .* eye(nbCoef); %init P matrix
end
    
tau = block.DialogPrm(1).Data;
alpha_z = block.DialogPrm(2).Data;
beta_z = block.DialogPrm(3).Data;
r = block.DialogPrm(4).Data;
lambda = block.DialogPrm(8).Data;

%input phase and reference
phi = block.InputPort(1).Data;
y_r = block.InputPort(2).Data(1);
dy_r = block.InputPort(2).Data(2);
ddy_r = block.InputPort(2).Data(3);

%weights
W = VonMises_Basis(phi, block.Dwork(nbFuncs+1).Data, block.Dwork(nbFuncs+2).Data); %use Von-Mises basis
W = W ./ sum(W,1);
%input polynomial
n=0:nbCoef-1;
poly_S = mod(phi, 2*pi).^(n'); %constrained states to 0~2pi

%Output force term data
F_d = (tau^2 * ddy_r - alpha_z*(beta_z*(0 - y_r) - tau*dy_r)) / r; %cancel amptitude r 

%Recursively update the weights
for k =1:nbFuncs
    w_k = block.Dwork(nbFuncs).Data;
    P_rlwr(:, :, k) = 1/lambda * (P_rlwr(:, :, k) - (P_rlwr(:, :, k)*(poly_S * poly_S' )*P_rlwr(:, :, k))/(lambda / W(k) + poly_S' * P_rlwr(:, :, k) * poly_S));
    error_k = F_d - w_k' * poly_S;
    block.Dwork(nbFuncs).Data = w_k + W(k) * P_rlwr(:, :, k) * poly_S * error_k;
end

%end Update

%%
%% Terminate:
%%   Functionality    : Called at the end of simulation for cleanup
%%   Required         : Yes
%%   C MEX counterpart: mdlTerminate
%%
function Terminate(block)

%end Terminate

%% Utility Functions (not call back)
