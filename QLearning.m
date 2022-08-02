function [Q, numSteps] = QLearning(GW, Q, params, states, goal, transitionList)
% QLEARNING runs the QLearning algorithm. It takes in a gridword, matrix of
% Q values, the goal state, the transitionList, and a vector
% of the possible actions.

numStates = params.numStates;
numActions = params.numActions;
actions = params.actions;
numSessions = params.numSessions;
numTrials = params.numTrials;
epsilon = params.epsilon;
gamma = params.gamma;
alpha = params.alpha;
session_rewards = zeros(1, numSessions);
numSteps = zeros(1, numSessions);

for ses = 1:numSessions
    
    % Start at random state for each session
    current = randi(numStates);
    cur_session_reward = 0;
    
    for trial = 1:numTrials
        prevState = current;

        % Choose action
        a = chooseAction(epsilon, actions, Q, prevState);

        % Take action a and move to next state by transitionList
        state = transitionList(prevState, a);
        % Observe reward
        reward = GW(state);
        cur_session_reward = cur_session_reward + reward;

        Q(prevState, a) = Q(prevState, a) + alpha * (reward + gamma * max(Q(state,:)) - Q(prevState, a));

        if state == goal
            break
        end

        numSteps(ses) = numSteps(ses) + 1;
        current = state;
    end
    session_rewards(ses) = cur_session_reward;

end
