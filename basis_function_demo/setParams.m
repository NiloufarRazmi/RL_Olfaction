function [params] = setParams(n_side)
% SETPARAMS sets the parameters for the gridworld. It outputs the
% parameters in a variable params.

params.rows = n_side;
params.cols = n_side;
params.GW_Size = params.cols * params.rows;
% NumStates is rows*cols for one state per grid loc for each context
params.numStates = params.rows * params.cols * params.numContexts;


params.actions = ['U'; 'D'; 'L'; 'R'];
params.numActions = 4;

% True for walls, false for continuous
params.walls = true;




end