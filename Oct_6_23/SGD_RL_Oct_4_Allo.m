function[output] = SGD_RL_Oct_4_Allo(subID)


rng('shuffle')


%% Initialize:
numContexts = 4; % 4 different contexts : 2 different lights - 2 different odors

% define the parameters of the gridworld
params = setParams(numContexts);
params.condition = 2; % 1 = single odor condition - 2 = two-odor condition
params.numSessions = 100;
numRuns = params.numSessions; % the number of sessions

numReps = 1;
states_n = params.numStates * numContexts ; % number of states
action_n = length(params.numActions); % number of actions

reward_port_1 = 75; % Right port odor 1 = rewarded
terminal_port_2 = 51; % Right port odor 2 = episode ends with no reward

reward_port_2 = 76; % Left port odor 2 = rewarded
terminal_port_1 = 100; % Left port odor 1 = episode ends with no reward


params.start = randi(params.numStates/2,1,1);

% Build the grid world enviroment
[GW, Q, states, transitionList,odor_port,reward_id] = buildGrid(params);


% Simulate episodes:
 input_state = convert_state(params.GW_Size,2,2);
 input_state = input_state./(sum(input_state(:,1)));
 
%input_state = eye(100,100);
numSamples_batch = 60000;
epsilon = 0.2;
gamma = 0.8;
actions = params.actions;

n_input_layer = size(input_state,1);
n_hidden_layer_1 = 100; % Number of neurons in the first hidden layer
n_hidden_layer_2 = 50; % Number of neurons in the second hidden layer
n_output_layer = 4;

weights_1_2 = randn(n_hidden_layer_1,n_input_layer)*sqrt(2/n_input_layer);
weights_2_3  = randn(n_hidden_layer_2,n_hidden_layer_1)*sqrt(2/n_hidden_layer_1);
weights_3_4  = randn(n_output_layer,n_hidden_layer_2)*sqrt(2/n_hidden_layer_2);

bias_1_2 = randn(n_hidden_layer_1,1);
bias_2_3 = randn(n_hidden_layer_2,1);
bias_3_4 = randn(n_output_layer,1);

n_iterations = 10;
accuracy = zeros(n_iterations,1);
random_accuracy = accuracy;


for iter=1:n_iterations

% store {(s_i,a_i,s'_i,r_i)} for a mini-batch:
batch_samples = zeros(numSamples_batch,5);

%% Run Experiment:

% set the expolation factor 
if iter > 4
    epsilon = 0.2;
else

    epsilon = 1;
end

[GW, Q, states, transitionList,odor_port,reward_id] = buildGrid(params);

% set initial state randomly:
current  = (params.numStates/4) * (odor_port-1) + randi((params.numStates/4),1,1);

for i= 1:numSamples_batch

    prevState = current;

    a1 = input_state(:,prevState);
    z2 =  weights_1_2*a1 + bias_1_2;
    a2 = Relu_func(z2);
    z3 = weights_2_3*a2 + bias_2_3;
    a3 = Relu_func(z3);
    z4 = weights_3_4*a3 + bias_3_4;
    a4 = Relu_func(z4); %Output vector

    % the output should be the activity of the last layer aka a 4*1
    % vector of action values:

    state_network_ouput = a4;

    % Choose action -- epsilon greedy (actions could be selected via softmax
    % on output neuron activity ???)
    a = chooseAction_NN(epsilon, actions, state_network_ouput);

    % Take action a and move to next state by transitionList
    state = transitionList(prevState, a);


    if prevState == reward_port_1 % reward id = 1 , odor_port =1
        reward = 100;
                params.start = randi(params.numStates/2,1,1);
        [GW, Q, states, transitionList,odor_port,reward_id] = buildGrid(params);
        current  = (params.numStates/4) * (odor_port-1) + randi((params.numStates/4),1,1);

    elseif prevState == reward_port_2 % reward id = 2 , odor_port = 2
        reward = 100;
                params.start = randi(params.numStates/2,1,1);
        [GW, Q, states, transitionList,odor_port,reward_id] = buildGrid(params);
        current  = (params.numStates/4) * (odor_port-1) + randi((params.numStates/4),1,1);

    elseif prevState== terminal_port_2 % reward id = 1 , odor_port = 2
        reward = 0;
                params.start = randi(params.numStates/2,1,1);
        [GW, Q, states, transitionList,odor_port,reward_id] = buildGrid(params);
        current  = (params.numStates/4) * (odor_port-1) + randi((params.numStates/4),1,1);

    elseif prevState== terminal_port_1 % reward id = 2 , odor_port = 1
        reward = 0;
                params.start = randi(params.numStates/2,1,1);
        [GW, Q, states, transitionList,odor_port,reward_id] = buildGrid(params);
        current  = (params.numStates/4) * (odor_port-1) + randi((params.numStates/4),1,1);
    else
        reward = 0;
        current = state;

    end

    % Store sample
    batch_samples (i,1) = prevState;
    batch_samples (i,2) = a;
    batch_samples (i,3) = state;
    batch_samples (i,4) = reward;

end


accuracy(iter) = sum(batch_samples(:,4));
avg_n_steps(iter) = numSamples_batch./(accuracy(iter)/max(batch_samples (:,4)));
random_accuracy(iter) = sum(batch_samples(:,4));

%% Learn :

%learning rate
eta = 0.00001;

%Initializing errors and gradients
error4 = zeros(n_output_layer,1);
error3 = zeros(n_hidden_layer_2,1);
error2 = zeros(n_hidden_layer_1,1);
errortot4 = zeros(n_output_layer,1);
errortot3 = zeros(n_hidden_layer_2,1);
errortot2 = zeros(n_hidden_layer_1,1);

grad4 = zeros(n_output_layer,1);
grad3 = zeros(n_hidden_layer_2,1);
grad2 = zeros(n_hidden_layer_1,1);
epochs =200;



m = 10; % Minibatch size

for k = 1:epochs % Outer epoch loop
    batches = 1;
    total_reward = 0;

    for j = 1:numSamples_batch/m

        error4 = zeros(n_output_layer,1);
        error3 = zeros(n_hidden_layer_2,1);
        error2 = zeros(n_hidden_layer_1,1);

        errortot4 = zeros(n_output_layer,1);
        errortot3 = zeros(n_hidden_layer_2,1);
        errortot2 = zeros(n_hidden_layer_1,1);

        grad4 = zeros(n_output_layer,1);
        grad3 = zeros(n_hidden_layer_2,1);
        grad2 = zeros(n_hidden_layer_1,1);

    for i = batches:batches+m-1 %Loop over each minibatch
    
    %Feed forward
    a1 = input_state(:,batch_samples(i,1));
    z2 = weights_1_2*a1 + bias_1_2;
    a2 = Relu_func(z2);
    z3 = weights_2_3*a2 + bias_2_3;
    a3 = Relu_func(z3);
    z4 = weights_3_4*a3 + bias_3_4;
    a4 = Relu_func(z4); %Output vector


    state_network_ouput = a4;

    % Choose action -- epsilon greedy (actions could be selected via softmax
    % on output neuron activity ???)
    a = batch_samples(i,2);

    % Take action a and move to next state by transitionList
    state = batch_samples(i,3);

    reward = batch_samples(i,4);

     %Feed forward again:
    a1_new = input_state(:,state);
    z2_new = weights_1_2*a1_new + bias_1_2;
    a2_new = Relu_func(z2_new);
    z3_new = weights_2_3*a2_new + bias_2_3;
    a3_new = Relu_func(z3_new);
    z4_new = weights_3_4*a3_new + bias_3_4;
    
    %backpropagation
    target = reward + gamma * max(elu(z4_new));
    error_action =(a4(a)-target).*Relu_func_dx(z4(a));
    error4 = zeros(length(z4),1);
    error4(a) = error_action;
    error3 = (weights_3_4'*error4).*Relu_func_dx(z3);
    error2 = (weights_2_3'*error3).*Relu_func_dx(z2);
    
    errortot4 = errortot4 + error4;
    errortot3 = errortot3 + error3;
    errortot2 = errortot2 + error2;


    grad4 = grad4 + error4*a3';
    grad3 = grad3 + error3*a2';
    grad2 = grad2 + error2*a1';

    total_reward = total_reward + reward;

    end
    
    %Gradient descent
    weights_3_4 = weights_3_4 - eta/m*grad4;
    weights_2_3 = weights_2_3 - eta/m*grad3;
    weights_1_2 = weights_1_2 - eta/m*grad2;

    bias_3_4 = bias_3_4 - eta/m*errortot4;
    bias_2_3 = bias_2_3 - eta/m*errortot3;
    bias_1_2 = bias_1_2 - eta/m*errortot2;
    batches = batches + m;

    end
    all_errors(k) = sum(errortot4);
    fprintf('Epochs:');
    %disp(k) %Track number of epochs
    [~, x] = sort(rand(1,numSamples_batch)); % shuffle order of stored dataset
    batch_samples = batch_samples(x,:);

end
end

%% Save results
output.weights_3_4 = weights_3_4;
output.weights_2_3 = weights_2_3;
output.weights_1_2 = weights_1_2;

output.bias_3_4 = bias_3_4;
output.bias_2_3 = bias_2_3;
output.bias_1_2 = bias_1_2;

output.accuracy = accuracy;
output.random_accuracy = random_accuracy;
output.subID = subID;

output.avg_n_steps =avg_n_steps;

t = rem(now,1) * 100000;
filename = [int2str(t),'_',int2str(subID),'_file'];
save(filename,'output')