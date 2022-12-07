
% Policy Evaluation:
function [state_values] = policy_eval(input)

for i=1:input.num_states
    delta = inf;
    while delta > 0.01
        old_value = input.state_values(i);
        input.state_values(i) = 0;
        for j=1:input.num_actions
            next_state = input.action_trans_function(i,j);
            reward = input.reward_function(next_state);
            input.state_values(i) = input.state_values(i) + input.policy(i,j) * (reward + input.gamma * input.state_values(next_state));
        end
        delta = min(delta,abs(input.state_values(i) - old_value));
    end
end

state_values = input.state_values;
end