function [policy] = mdp_softmax(beta, action_values)

% softmax action selection -- 
% Beta = temperature
% action_values = a vector of action values
% output: probability vector of choosing each action

policy = exp(action_values./beta)./(sum(exp(action_values./beta)));
