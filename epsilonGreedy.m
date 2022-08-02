function [chosenAction] = epsilonGreedy(epsilon, actions, Q, state)
% EPSILONGREEDY takes in a value epsilon between 0 and 1, a list of
% action, a matrix of Q values, and a state. It outputs the chosen action.
% 1 = 'U', 2 = 'D', 3 = 'R', 4 = 'L'
    if rand <= epsilon
        % Choose random action
        chosenAction = randsample(1:length(actions), 1);
    else
        max_a = find(Q(state,:)==max(Q(state, :)));
        % Choose max Q value
        chosenAction = max_a(randi(length(max_a)));
    end
end