function [chosenAction] = chooseAction(epsilon, actions, Q, state)
%CHOOSEACTION Choose action based on action-selection method

chosenAction = epsilonGreedy(epsilon, actions, Q, state);

end

