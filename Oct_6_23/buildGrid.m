function [GW, Q, states, transitionList,odor_port,reward_id] = buildGrid(params)
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
transitionList = alloTransitionList(params, states, walls);
a = transitionList;

if params.arena ==1 
     if rand<0.5
            % Go to the context with rewarding state on the right
            transitionList(21,1) = 71;
            reward_id = 1;
            odor_port = 1;

        else
            % Go to the context with rewarding state on the left
            transitionList(21,1) = 80;
            reward_id = 2;
            odor_port = 1;

     end
else



% If odor is on top:
if start<26
    if num_odor==1 % single odor condition
        transitionList(21,1) = 71;
        reward_id = 1;
        odor_port = 1;

    else % two-odor condition
        if rand<0.5
            % Go to the context with rewarding state on the right
            transitionList(21,1) = 71;
            reward_id = 1;
            odor_port = 1;

        else
            % Go to the context with rewarding state on the left
            transitionList(21,1) = 80;
            reward_id = 2;
            odor_port = 1;

        end
    end
else
    if num_odor==1 % single odor condition
        transitionList(30,1) = 71;
        reward_id = 1;
                odor_port = 2;


    else % two-odor condition
        if rand<0.5
            transitionList(30,2) = 71;
            reward_id = 1;
            odor_port = 2;
        else
            transitionList(30,2) = 80;
            reward_id = 2;
            odor_port = 2;

        end
    end
end
end


end
