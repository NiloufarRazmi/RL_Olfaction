function [params] = setParams(numContexts)
% SETPARAMS sets the parameters for the gridworld. It outputs the
% parameters in a variable params.

params.numContexts = numContexts;
params.contexts = {};
params.rows = 5;
params.cols = 5;

% NumStates is rows*cols for one state per grid loc for each context
params.numStates = params.rows * params.cols * params.numContexts;

params.actions = ['U'; 'D'; 'L'; 'R'];
params.numActions = 4;

% True for walls, false for continuous
params.walls = true;

% Action-selection parameters
params.epsilon = 0.1;

% QLearning parameters
params.gamma = 0.8;
params.alpha = 0.05;

% Set up the task
params.numTrials = 1000;
params.numSessions = 200;

params.wallsLoc = [1,2,4,5,6,16,21,22,24,25,20,10];

end