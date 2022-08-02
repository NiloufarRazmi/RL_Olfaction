function [transitionList] = createTransitionList(params, states, walls)
% CREATETRANSITIONLIST takes in params, a matrix of states, and whether
% there are walls (true or false). The output is a state action state
% transition list. Each row is a state and the columns are the resulting
% states of taking the four actions (1 = 'U', 2 = 'D', 3 = 'R', 4 = 'L').


transitionList = zeros(params.numStates, 4);

% Case when bouncing back off of walls
if walls
    for context = 1:params.numContexts
        for state = params.contexts{context}
            [row, col] = find(states == state);

            % Up column
            if row == 1
                transitionList(state, 1) = state;
            else
                transitionList(state, 1) = states(row - 1, col);
            end

            % Down column
            if row == params.rows
                transitionList(state, 2) = state;
            else
                transitionList(state, 2) = states(row + 1, col);
            end

            % Right column
            if col == params.cols || col == params.cols*params.numContexts
                transitionList(state, 3) = state;
            else
                transitionList(state, 3) = states(row, col + 1);
            end

            % Left column
            if col == 1 || col == params.cols+1
                transitionList(state, 4) = state;
            else
                transitionList(state, 4) = states(row, col - 1);
            end
        end
    end
% Case when continuing around when on edge
else
    % FIX THIS FIX THIS right for rightmost column to be all mods and more
    % general
    for context = 1:params.numContexts
        for state = params.contexts{context}
            [row, col] = find(states == state);

            % Up column
            if row == 1
                transitionList(state, 1) = states(params.rows, col);
            else
                transitionList(state, 1) = states(row - 1, col);
            end

            % Down column
            if row == params.rows
                transitionList(state, 2) = states(1, col);
            else
                transitionList(state, 2) = states(row + 1, col);
            end

            % Right column
            if mod(col, params.cols) == 0
                transitionList(state, 3) = states(row, col - (params.cols-1));
            else
                transitionList(state, 3) = states(row, col + 1);
            end

            % Left column
            if mod(col, params.cols) == 1
                transitionList(state, 4) = states(row, col + params.cols - 1);
            else
                transitionList(state, 4) = states(row, col - 1);
            end
        end
    end
end

