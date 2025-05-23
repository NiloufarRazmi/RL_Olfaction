function [Q, numSteps, w,output] = QLearningFuncApproxEgo( GW, Q, params, states, transitionList, w, x)
% QLEARNINGFUNCAPPROX runs the QLearning algorithm with function 
% approximation. It takes in a gridword, matrix of Q values, 
% the goal state, the transitionList, a matrix of weights, and a matrix of 
% feature vectors.

numContexts = params.numContexts;
numStates = params.numStates;
actions = params.actions;
numSessions = params.numSessions;
epsilon = params.epsilon;
gamma = params.gamma;
alpha = params.alpha;
numSteps = zeros(1, numSessions);
state_occupancy =  zeros(size(GW)); 

%generating standardized starting point sequence
%start = csvread('starting_locations.csv');

for ses = 1:numSessions
  
    params.start = randi(200,1,1);
    state_mov =[];
    [~, ~, ~, transitionList,odor_id] = buildGridEgo(params);
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
        % based on the odor identity (but they are not simulatneously present)
        if state ==273
            reward = 10;
        elseif state == 378
            reward = 10;
        else
            reward = 0;
        end
  
        
        % Update weights
        w(:, a) = w(:, a)' + alpha ./ (size(x,1)./(size(x,2)./2)) * (reward + gamma * max(Q(state, :)) - Q(prevState, a)) * (x(:, prevState)');
        
        % store things for plotting
        state_mov = [state_mov,state];
        state_occupancy(state) = state_occupancy(state)+1;
        
        % if reward is obtained , the episode is done
        if prevState == 273 && a == 1
            break
        elseif prevState == 378 && a==1
            break
        end
        numSteps(ses) = numSteps(ses) + 1;
        output.odor_id(ses) = odor_id;
        output.port_id(ses) =params.start;
        output.state_occupancy(ses,:) = state_occupancy;

        current = state;
%            if numSteps(ses) > 20000
%                break
%            end
    end
    
    output.state_mov{ses} = state_mov;

    
end