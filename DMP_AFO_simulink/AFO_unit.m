function AFO_unit(block)
% level2 s-function of simple adaptive oscillator unit
% input: e(t); 
% output: phi, omega, alpha, alpha*cos(phi)
% parameters: K, eta, phi0, omega0, alpha0
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
block.NumInputPorts  = 1;
block.NumOutputPorts = 4;

% Setup port properties to be inherited or dynamic
block.SetPreCompInpPortInfoToDynamic;
block.SetPreCompOutPortInfoToDynamic;

% Override input port properties
%input1: e(t)
block.InputPort(1).Dimensions        = 1;
block.InputPort(1).DatatypeID  = 0;  % double
block.InputPort(1).Complexity  = 'Real';
block.InputPort(1).DirectFeedthrough = true;

% Override output port properties
%phi
block.OutputPort(1).Dimensions       = 1;
block.OutputPort(1).DatatypeID  = 0; % double
block.OutputPort(1).Complexity  = 'Real';
%omega
block.OutputPort(2).Dimensions       = 1;
block.OutputPort(2).DatatypeID  = 0; % double
block.OutputPort(2).Complexity  = 'Real';
%alpha
block.OutputPort(3).Dimensions       = 1;
block.OutputPort(3).DatatypeID  = 0; % double
block.OutputPort(3).Complexity  = 'Real';

%alpha*cos(phi)
block.OutputPort(4).Dimensions       = 1;
block.OutputPort(4).DatatypeID  = 0; % double
block.OutputPort(4).Complexity  = 'Real';

block.OutputPort(1).SamplingMode = 'Sample';
block.OutputPort(2).SamplingMode = 'Sample';
block.OutputPort(3).SamplingMode = 'Sample';
block.OutputPort(4).SamplingMode = 'Sample';

% Register parameters
block.NumDialogPrms     = 5; % parameters: K, eta, phi0, omega0, alpha0

% Continuous states
block.NumContStates = 3; %no discrete states, don't setup in 'DoPostPropSetup'

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

% block.RegBlockMethod('PostPropagationSetup',    @DoPostPropSetup); %no discrete states
block.RegBlockMethod('InitializeConditions', @InitializeConditions);
block.RegBlockMethod('Derivatives', @Derivatives);  %update derivatives of continuous states
% block.RegBlockMethod('Start', @Start);
block.RegBlockMethod('Outputs', @Outputs);     % Required
% block.RegBlockMethod('Update', @Update);
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
% parameters: K, eta, phi0, omega0, alpha



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
% parameters: K, eta, phi0, omega0, alpha
%%Continuous states
block.ContStates.Data(1) = block.DialogPrm(3).Data; %phi_0
block.ContStates.Data(2) = block.DialogPrm(4).Data; %omega_0
block.ContStates.Data(3) = block.DialogPrm(5).Data; %alpha_0

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
%Outputs: continuous states
block.OutputPort(1).Data = block.ContStates.Data(1); % phi
block.OutputPort(2).Data = block.ContStates.Data(2); % omega
block.OutputPort(3).Data = block.ContStates.Data(3); % alpha
block.OutputPort(4).Data = block.ContStates.Data(3) * cos(block.ContStates.Data(1)); % alpha*cos(phi)
%end Outputs

%% Derivatives:
%%   Functionality    : update derivatives of continuous states
%%                                  (when continuous states are used)
%%   Required         : No
function Derivatives(block)
K = block.DialogPrm(1).Data;
eta =  block.DialogPrm(2).Data;
error = block.InputPort(1).Data;

if abs(error)<1e-2
    error =0;
end

% if block.ContStates.Data(2)<0
%     block.ContStates.Data(2) = 0;
% end

% update dPhi
block.Derivatives.Data(1) = block.ContStates.Data(2) - K*error*sin(block.ContStates.Data(1));  
% update dOmega
block.Derivatives.Data(2) = -K*error*sin(block.ContStates.Data(1));
% update dalpha
block.Derivatives.Data(3) = eta*error*cos(block.ContStates.Data(1));
%end Derivatives

%%
%% Update:
%%   Functionality    : Called to update discrete states
%%                      during simulation step
%%   Required         : No
%%   C MEX counterpart: mdlUpdate
%%
function Update(block)

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
