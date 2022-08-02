function [Q, numSteps, w,state_occupancy,state_mov] = QLearningFuncApprox( GW, Q, params, states, transitionList, w, x)
% QLEARNINGFUNCAPPROX runs the QLearning algorithm with function 
% approximation. It takes in a gridword, matrix of Q values, 
% the goal state, the transitionList, a matrix of weights, and a matrix of 
% feature vectors.

numContexts = params.numContexts;
numStates = params.numStates;
actions = params.actions;
numSessions = 200;
epsilon = params.epsilon;
gamma = params.gamma;
alpha = params.alpha;
numSteps = zeros(1, numSessions);
state_occupancy =  zeros(size(GW)); 

%generating standardized starting point sequence
%start = csvread('starting_locations.csv');

for ses = 1:numSessions
  
    params.start = randi(50,1,1);
    state_mov =[];
    [~, ~, ~, transitionList] = buildGrid(params);
    current = params.start ;

    while true

        prevState = current;
        % Value function s, a
        Q = (w'*x)';
        % Choose action
        a = chooseAction(epsilon, actions, Q, prevState);

        % Take action a and move to next state by transitionList
        state = transitionList(prevState, a);

        % Observe reward -- curently there are two rewarding actions
        % based on the odor location (but they are not simulatneously present)
        if prevState == 73 && a == 3
            reward = 10;
        elseif prevState == 78 && a==4
            reward = 10;
        else
            reward = 0;
        end
  
        % Update weights
        w(:, a) = w(:, a)' + alpha ./ (size(x,1)./(size(x,2)./2)) * (reward + gamma * max(Q(state, :)) - Q(prevState, a)) * (x(:, prevState)');
        

        state_mov = [state_mov,state];
        state_occupancy(state) = state_occupancy(state)+1;

        if prevState == 73 && a == 3
            break
        elseif prevState == 78 && a==4
            break
        end
        numSteps(ses) = numSteps(ses) + 1;
        current = state;
%           if numSteps(ses) > 20000
%               break
%           end
    end
    
end