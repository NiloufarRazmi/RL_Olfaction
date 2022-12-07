

% States = { N_1 , S_1 , E_1 , W_1 , N_2, S_2, E_2 ,W_2}
% N,S,E,W_1 before smelling the odor, N,S,E,W_2 the same ports after
% smelling the odor. 

% Actions = { N , S , E , W}

input.num_states = 8;
input.num_actions=4;
input.reward_function = zeros(1,input.num_states+1); %State 9 is getting the reward.
input.reward_function(end) = 10;
input.state_values=zeros(1,input.num_states+1); % State 9 is getting the reward.
input.gamma = 0.8; % discount factor
beta = 2; % temperature of softmax function

% Assign the reward port
if rand<0.5
    r = 3; % reward east
else
    r = 4; % reward west
end

% Assign odor port:
if rand<0.5
    input.action_trans_function = transition_function(input.num_states,1,r);
else
    input.action_trans_function = transition_function(input.num_states,2,r);
end


%policy = 0.25 * ones(size(action_trans_function));

% a random policy
%input.policy = randi(4,size(input.action_trans_function,1),1);
input.policy = 0.25 * ones(size(input.action_trans_function));


input.state_values = policy_eval(input);
policy_stable = false;


% Policy Improvement:

while ~policy_stable
action_set_values = zeros(input.num_actions,1);

policy_stable = true;

for i=1:input.num_states
        old_action = input.policy(i,:);
        for j = 1: input.num_actions
            next_state= input.action_trans_function(i,j);
            reward = input.reward_function(next_state);
            action_set_values(j) = input.policy(i,j) *(reward + input.gamma * input.state_values(next_state));
        end
        input.policy(i,:) = mdp_softmax(beta,action_set_values);
        if old_action~=input.policy(i,:)
            policy_stable =  false;
        end
end

if policy_stable
    break
else
    input.state_values = policy_eval(input);
end

end

