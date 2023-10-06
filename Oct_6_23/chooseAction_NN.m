function [chosenAction] = chooseAction_NN(epsilon, actions, Q)
%CHOOSEACTION Choose action based on action-selection method

chosenAction = epsilonGreedy_NN(epsilon, actions, Q);

end

