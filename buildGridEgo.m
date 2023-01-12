function [GW, Q, states, transitionList,odor_id] = buildGridEgo(params)
% BUILDGRID builds the grid for gridworld using the params and the goal.
% It outputs a gridworld, matrix of Q values, a matrix of states,
% the goal state, and a transitionList.

numContexts = params.numContexts;
numStates = params.numStates;
rows = params.rows;
cols = params.cols;
walls = params.walls;
totNumStates = numStates * numContexts;
start = params.start;
num_odor= params.condition;

% Build grid of zeros, true values
GW = zeros(totNumStates,1);

% To store action-value estimates 
Q = zeros(totNumStates, params.numActions);

% Create matrix of states 1 ... n
states = reshape(1:25, rows, cols);

% Create transition list
transitionList = egoTransitionList(params, states, walls);
a = transitionList;

transitionList = [transitionList ; a+numStates ;a+numStates*2 ;a+numStates*3];

% assign odor identity:

if start<101  % top port active ( we go state 11 facing north)
    % randomly assign odor 1 or
    if num_odor==1 % single odor condition
        transitionList(12,1) = 211;
        odor_id = 1;
    else % two-odor condition
        if rand < 0.5
            transitionList(12,1) = 211;
            odor_id = 1;
        else % odor 2
            transitionList(12,1) = 311;
            odor_id = 2;

        end
    end

else %  bottom port active - ( we go to state 15 facing south)
    %randomly assign odor 1

    if num_odor ==1 % single odor condition
        transitionList(139,1) = 240;
        odor_id = 1;
    else  % two-odor condition

        if rand<0.5
            transitionList(139,1) = 240;
            odor_id = 1;
        else
            % or odor 2
            transitionList(139,1) = 340;
            odor_id = 2;
        end

    end

end


end
