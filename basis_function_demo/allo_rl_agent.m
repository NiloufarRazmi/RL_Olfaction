% Allocentric version of the RL navigation task in an open arena with 2
% odors

%% initialize :
n_lights = 2;
n_odors = 2;

% define the parameters of the gridworld:
numContexts = n_odors + n_lights; % 4 different contexts : 2 different lights - 2 different odors
params.n_side = 5;
params.numContexts = numContexts;
params.rows = params.n_side;
params.cols = params.n_side;
params.GW_Size = params.cols * params.rows;
params.numStates = params.rows * params.cols * params.numContexts;
params.actions = ['U'; 'D'; 'L'; 'R'];
params.numActions = size(params.actions,1);
% True for walls, false for continuous
params.walls = true;
params.arena = 1; % 1 triangle , 2 diamond arena
params.condition = n_odors; % number of odors

n_place = params.GW_Size;
n_states = n_place * n_odors * n_lights ; % number of total states
num_Reps = 50; % number of simulations of an experiment
numSamples_batch = 1000; % number of steps in the arena
batch_samples = zeros(numSamples_batch,4);
total_reward = zeros(num_Reps,2);


for rep_cond  = 1:2

    % state represntation:

    % In this part, we are trying to create the input layer for our
    % Q-learning algorithm. Input (also referred to as "neurons" below)
    % are the states that our RL agent learns with. These states could be
    % defined in different ways.

    % based on rep_cond define the input representation as either 1) joint representation of light/place and odor/place 

    % 2) seperate representation of both odor/place and light/plce

    if rep_cond == 1  % joint representation

        x = eye(n_states,n_states);% input layer is just the same as states

    elseif rep_cond ==2 %  seperate place,light and odor representation:

        x = convert_state(n_place,n_odors,n_lights);
    end


    % Q values for the states of the agents
    totalQ = zeros(size(x,2),params.numActions);
    w = zeros(size(x,1),params.numActions);

    %% Initialize:

    reward_port_1 = params.GW_Size * 3; % Right port odor 1 = rewarded
    terminal_port_2 = params.GW_Size * 2 + 1; % Right port odor 2 = episode ends with no reward

    reward_port_2 = params.GW_Size * 3 +1; % Left port odor 2 = rewarded
    terminal_port_1 = params.GW_Size * 4; % Left port odor 1 = episode ends with no reward

  

    % Build the grid world enviroment
    [GW, Q, states, transitionList,odor_port,reward_id] = buildGrid(params);


    % set initial state randomly:

    %% Run the experiment :

    % random initial point:
    A = reshape(1:params.GW_Size,params.n_side,params.n_side);
    U = triu(A);
    U = U(:);
    U(U==0) =[];
    init_states = U;
    idx = randi(length(init_states),1,1);
    current  = init_states(idx);

    % Walls and blocked states:
    A(diag(A)) = 0;
    L = tril(A);
    L = L(:);
    L(L==0) =[];
    params.blockstates = L;
    all_blockStates = [params.blockstates; params.blockstates+params.GW_Size*2;params.blockstates+params.GW_Size*3];

    epsilon = 0.2; % exploration factor
    actions = params.actions;
    alpha= 0.005; % learning rate
    gamma = 0.8; % discount factor


    for  r = 1: num_Reps

        for i= 1:numSamples_batch

            prevState = current;
            Q = (x(:,prevState)'* w);

            % Choose action -- epsilon greedy (actions could be selected via softmax
            % on output neuron activity ???)
            a = chooseAction_NN(epsilon, actions, Q);


            % Take action a and move to next state by transitionList
            state = transitionList(prevState, a);

            % If encounter blocks stay where you are
            if ismember(state,params.blockstates)
                state = prevState;
            end

            next_Q = (x(:,state)'*w);

            % Check if you've reached a terminal state
            if prevState == reward_port_1 % reward id = 1 , odor_port =1
                reward = 100;
                [GW, Q, states, transitionList,odor_port,reward_id] = buildGrid(params);
                idx = randi(length(init_states),1,1);
                current  = init_states(idx);
            elseif prevState == reward_port_2 % reward id = 2 , odor_port = 2
                reward = 100;
                [GW, Q, states, transitionList,odor_port,reward_id] = buildGrid(params);
                idx = randi(length(init_states),1,1);
                current  = init_states(idx);
            elseif prevState== terminal_port_2 % reward id = 1 , odor_port = 2
                reward = 0;
                [GW, Q, states, transitionList,odor_port,reward_id] = buildGrid(params);
                idx = randi(length(init_states),1,1);
                current  = init_states(idx);
            elseif prevState== terminal_port_1 % reward id = 2 , odor_port = 1
                reward = 0;
                [GW, Q, states, transitionList,odor_port,reward_id] = buildGrid(params);
                idx = randi(length(init_states),1,1);
                current  = init_states(idx);
            else
                reward = 0;
                current = state;

            end

            % Update Weights:
            w(:, a) = w(:, a) + alpha ./ (size(x,1)./(size(x,2)./2)) * (reward + gamma * max(next_Q) - Q( a)) * x(:,prevState);

            % Store sample
            batch_samples (i,1) = prevState;
            batch_samples (i,2) = a;
            batch_samples (i,3) = state;
            batch_samples (i,4) = reward;

        end

        total_reward(r,rep_cond) = sum(batch_samples (:,4));

    end
end

plot(total_reward,'LineWidth',3)
legend('Cond = Conjunctive Rep','Cond = Seperate Rep')

