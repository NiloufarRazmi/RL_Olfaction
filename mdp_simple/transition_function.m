
function[trans_function] = transition_function(num_states,odor_port,reward_port)

% transition function for port-based mdp :
% States = { N_1 , S_1 , E_1 , W_1 , N_2, S_2, E_2 ,W_2}
% Actions = { N , S , E , W}
% taking action "a" in the corresponding states means "poke" (e.g. N in
% N_1)

trans_function = zeros(num_states,4);

for i=1:4
    trans_function(i,:) = [1 2 3 4];
end

for i=5:8
    trans_function(i,:) = [5 6 7 8];
end

% poking in the odor_port state transition you to the post_odor states:
trans_function(odor_port,odor_port) = odor_port+4;

% poking in the reward port transition you to state 9:
trans_function(reward_port+4,reward_port) = num_states+1;

end