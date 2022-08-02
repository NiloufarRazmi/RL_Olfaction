function [GW, Q, states, transitionList] = buildGrid(params)
% BUILDGRID builds the grid for gridworld using the params and the goal.
% It outputs a gridworld, matrix of Q values, a matrix of states,
% the goal state, and a transitionList.

numContexts = params.numContexts;
numStates = params.numStates;
rows = params.rows;
cols = params.cols;
walls = params.walls;
totNumStates = 200;
start = params.start;

% Build grid of zeros, true values
GW = zeros(totNumStates,1);

% To store action-value estimates (4 for each basis function)
Q = zeros(totNumStates, 4);

% Create matrix of states 1 ... n
states = reshape(1:25, rows, cols*numContexts);

% Create transition list
transitionList = rotateTransitionList(params, states, walls);
a = transitionList;
transitionList = [transitionList ; a+25 ;a+50 ;a+75];

% If odor is on top:
if start<26
    if rand<0.5
    % Go to the context with rewarding state on the right
    transitionList(11,1) = 51;
    else
     % Go to the context with rewarding state on the left
    transitionList(11,1) = 76;  
    end
else
    if rand<0.5
    transitionList(40,2) = 51;
    else
    transitionList(40,2) = 76;  
    end
end


end
