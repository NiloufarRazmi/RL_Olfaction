
function[mutual_info] = mutual_info_analysis_allo(output)

%% Initialize:
numContexts = 4; % 4 different contexts : 2 different lights - 2 different odors

% define the parameters of the gridworld
params = setParams(numContexts);
params.condition = 2; % 1 = single odor condition - 2 = two-odor condition
params.numSessions = 100;
numRuns = params.numSessions; % the number of sessions
params.arena =2;

numReps = 1;
states_n = params.numStates ; % number of states
action_n = length(params.numActions); % number of actions

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
n_hidden_layer_1 = size(output.weights_1_2,1); % Number of neurons in the first hidden layer
n_hidden_layer_2 = size(output.weights_2_3,1); % Number of neurons in the second hidden layer
n_output_layer = size(output.weights_3_4,1);


weights_1_2 = output.weights_1_2;
weights_2_3  = output.weights_2_3;
weights_3_4  = output.weights_3_4;

bias_1_2 = output.bias_1_2;
bias_2_3 = output.bias_2_3;
bias_3_4 = output.bias_3_4;

n_iterations = 15;
accuracy = zeros(n_iterations,1);
random_accuracy = accuracy;

batch_samples = zeros(numSamples_batch,5);

%% Run Experiment:

epsilon = 0.2;


[GW, Q, states, transitionList,odor_port,reward_id] = buildGrid(params);
% set initial state randomly:
current  = (params.numStates/4) * (odor_port-1) + randi((params.numStates/4),1,1);

for i= 1:numSamples_batch

    prevState = current;

    a1 = input_state(:,prevState);
    z2 =  weights_1_2*a1 + bias_1_2;
    a2 = elu(z2);
    z3 = weights_2_3*a2 + bias_2_3;
    a3 = elu(z3);
    z4 = weights_3_4*a3 + bias_3_4;
    a4 = elu(z4); %Output vector

    % the output should be the activity of the last layer aka a 4*1
    % vector of action values:

    state_network_ouput = a4;

    % Choose action -- epsilon greedy (actions could be selected via softmax
    % on output neuron activity ???)
    a = chooseAction_NN(epsilon, actions, state_network_ouput);

    % Take action a and move to next state by transitionList
    state = transitionList(prevState, a);


    if prevState == 75 % reward id = 1 , odor_port =1
        reward = 100;
                params.start = randi(params.numStates/2,1,1);
        [GW, Q, states, transitionList,odor_port,reward_id] = buildGrid(params);
        current  = (params.numStates/4) * (odor_port-1) + randi((params.numStates/4),1,1);

    elseif prevState == 76 % reward id = 2 , odor_port = 2
        reward = 100;
                params.start = randi(params.numStates/2,1,1);
        [GW, Q, states, transitionList,odor_port,reward_id] = buildGrid(params);
        current  = (params.numStates/4) * (odor_port-1) + randi((params.numStates/4),1,1);

    elseif prevState==51 % reward id = 1 , odor_port = 2
        reward = 0;
                params.start = randi(params.numStates/2,1,1);
        [GW, Q, states, transitionList,odor_port,reward_id] = buildGrid(params);
        current  = (params.numStates/4) * (odor_port-1) + randi((params.numStates/4),1,1);

    elseif prevState==100 % reward id = 2 , odor_port = 1
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
    neuron_sample(i,:) = a3;


end





%% Calculate Mutual Information:


for i=1:states_n
    prob_state(i) = sum(batch_samples(:,1) == i) / length(batch_samples(:,1));
end

for i =1:n_hidden_layer_2
        [Y,E] = discretize(neuron_sample(:,i),29);
        for j =1: 29
        prob_activity(i,j) = (sum(Y==j))/length(batch_samples(:,1));
        end
end

A = [ batch_samples(:,1) , Y ];
AA = unique(A,"rows");

mutual_info = zeros(n_hidden_layer_2,1);
for j=1:n_hidden_layer_2
    
for i=1:states_n

    if prob_state(i) > 0
    activity = AA(AA(:,1) == i,2);
    if prob_activity(j,activity) == 0
        prob_activity(j,activity) = 0.00000001;
    end
    mutual_info(j,1) = mutual_info(j,1) +  prob_state(i) * log10(1/prob_activity(j,activity));
    end
end
end


%% Qualitative Analysis:
n_positions = 25;
 n_sensory_cue = 4;

for i=1:n_positions
    input_state = zeros(n_input_layer,1);
    input_state(i) = 0.5;
    a1 = input_state;
    z2 =  weights_1_2*a1 + bias_1_2;
    a2 = elu(z2);
    z3 = weights_2_3*a2 + bias_2_3;
    a3 = elu(z3);
    all_activity (i,:) = a3;
end


for i=1:n_sensory_cue
    input_state = zeros(n_input_layer,1);
    input_state(n_positions+i) = 0.5;
    a1 = input_state;
    z2 =  weights_1_2*a1 + bias_1_2;
    a2 = elu(z2);
    z3 = weights_2_3*a2 + bias_2_3;
    a3 = elu(z3);
    all_activity_sensory (i,:) = a3;
end
