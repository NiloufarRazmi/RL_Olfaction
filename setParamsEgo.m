function [params] = setParamsEgo(nLights,nOdors,numSessions)
% SETPARAMS sets the parameters for the gridworld. It outputs the
% parameters in a variable params.

params.numContexts = nOdors + nLights;
params.contexts = {};
params.rows = 5;
params.cols = 5;
params.condition = nOdors;
params.numSessions = numSessions;
head_directions = 4;

% NumStates is rows*cols for one state per grid loc for each context
params.numStates = params.rows * params.cols *head_directions;

params.actions = ['S'; 'L'; 'R'];
params.numActions = 3;

% True for walls, false for continuous
params.walls = true;

% Action-selection parameters
params.epsilon = 0.1;

% QLearning parameters
params.gamma = 0.8;
params.alpha = 0.05;

end