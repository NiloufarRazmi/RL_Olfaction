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
num_odor= params.condition;
n_side = params.n_side;
n_locations = n_side ^2;

% Build grid of zeros, true values
GW = zeros(totNumStates,1);

% To store action-value estimates 
Q = zeros(totNumStates, params.numActions);
% Create matrix of states 1 ... n

states = reshape(1:n_locations, rows, cols);

% Create transition list

transitionList = alloTransitionList(params, states, walls);
a = transitionList;

if params.arena ==1 
     if rand<0.5
            % Go to the context with rewarding state on the right
            transitionList(n_side*(n_side-1)+1,1) = 2 * n_locations + n_side*(n_side-1)+1;
            reward_id = 1;
            odor_port = 1;

        else
            % Go to the context with rewarding state on the left
            transitionList(n_side*(n_side-1)+1,1) = 3 * n_locations + n_side*(n_side-1)+1; % changed this from 80 to 96.
            reward_id = 2;
            odor_port = 1;

     end
else



% If odor is on top:
if start<n_locations+1
    if num_odor==1 % single odor condition
        transitionList(n_side*(n_side-1)+1,1) = 2 * n_locations + n_side*(n_side-1)+1;
        reward_id = 1;
        odor_port = 1;

    else % two-odor condition
        if rand<0.5
            % Go to the context with rewarding state on the right
            transitionList(n_side,1) = 2 * n_locations + n_side*(n_side-1)+1;
            reward_id = 1;
            odor_port = 1;

        else
            % Go to the context with rewarding state on the left
            transitionList(n_side,1) = 3 * n_locations + n_side;
            reward_id = 2;
            odor_port = 1;

        end
    end
else
    if num_odor==1 % single odor condition
        transitionList(n_locations + n_side,1) = 2 * n_locations + n_side*(n_side-1)+1;
        reward_id = 1;
                odor_port = 2;


    else % two-odor condition
        if rand<0.5
            transitionList(n_locations + n_side,2) = 2 * n_locations + n_side*(n_side-1)+1;
            reward_id = 1;
            odor_port = 2;
        else
            transitionList(n_locations + n_side,2) = 3 * n_locations + n_side;
            reward_id = 2;
            odor_port = 2;

        end
    end
end
end


end
