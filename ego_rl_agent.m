
% Egocentric version of the RL navigation task in an open arena with 2
% odors

% There are 400 states in total:

% - There are 100 spatial states : 25 for each gridcell in the arena and 4
% possible head direction

% - There are 2 possible light ports ( top - bottom) and 2 possible odor (
% odor 1 - odor 2)

%% initialize :
nLights = 2;
nOdors = 2;
numSessions = 200; % number of trials within each experiment

numContexts = nLights + nOdors; % how many difernt contexts i.e. light/odor 
params = setParamsEgo(nLights,nOdors,numSessions);
total_states = params.numStates * numContexts; % number of total states 

numReps = 1; % number of simulations of an experiment

% note that params.numStates is the number of spatial states.

% what kind of state representaion are we using: 1) joint 2) light/place
% joint , odor-place seperate 3 ) all seperate
rep_cond = 1; 

jointRep = false; % are we using joint representation or seperate reperesentation

makePlots = true;


%% state represntation:

    % In this part, we are trying to create the input layer for our
    % Q-learning algorithm. Input (also referred to as "neurons" below)
    % are the states that our RL agent learns with. These states could be
    % defined in different ways.

    % based on rep_cond define the input representation as either 1) joint representation of light/place and odor/place or 2)
    % joint representation of light/place but seperate representaiton of
    % odor/place
    % 3) seperate representation of both odor/place and light/plce

    if rep_cond == 1  % joint representation 

       all_neurons = eye(total_states,total_states); % create an identity matrix representing each state of the experiment
       
       % input layer is just the same as states

       x=all_neurons;

    elseif rep_cond ==2 % joint light represntation , seperate odor represtnation:

        % neurons that jointly represent place and light:
        n_place_light = params.numStates * nLights; % number of place_light neurons

        place_light_neurons = zeros(total_states,n_place_light);

        % the identity matrix represnt them because their represtnation is
        % joint:
        place_light_neurons(1:n_place_light,1:n_place_light) = eye(n_place_light,n_place_light);
       
         % neurons that represnt place and odor ( but seperately) :
         n_place_odor = params.numStates + params.numStates * nOdors; % total number of them

         % preallocate the matrix 
         place_odor_neurons = zeros(total_states,n_place_odor);

         % place neurons , their activity is the identity matrix
         place_odor_neurons(n_place_light+1:end,1:params.numStates) = repmat(eye(params.numStates),nOdors,1);
        
         % neurons for odor 1
         place_odor_neurons(n_place_light+1:n_place_light+params.numStates,params.numStates+1:params.numStates+params.numStates) = ones(params.numStates,params.numStates);
         
         if nOdors ==2
         % neurons for odor 2
         place_odor_neurons(n_place_light+params.numStates+1:end,params.numStates+params.numStates+1:end) = ones(params.numStates,params.numStates);
         end

         % Add place_odor and place_light neurons together to get the input
         % layer:
        x = [place_light_neurons, place_odor_neurons];

    else  % neurons represtnting place, odor and light seperately
       
    
         % number of neurons:
         n_neurons = params.numStates + params.numStates*numContexts;

         % preallocate the matrix 
         all_neurons = zeros(total_states,n_neurons);

         % place neurons , their activity is the identity matrix
         all_neurons(:,1:params.numStates) = repmat(eye(params.numStates),numContexts,1);
        
         % neurons for light 1
         n = params.numStates; % counting the number of neurons so far 
         s =params.numStates;  % counting the number of states so far 

         all_neurons(1:s,n+1:n+params.numStates) = ones(params.numStates,params.numStates);
         n = n+params.numStates;

         % neurons for light 2
         all_neurons(s+1:s+params.numStates,n+1:n+params.numStates) = ones(params.numStates,params.numStates);
         n = n+params.numStates;
         s = s + params.numStates;

         % neurons for odor 1 
         all_neurons(s+1:s+params.numStates,n+1:n+params.numStates) = ones(params.numStates,params.numStates);
         n = n+params.numStates;
         s = s + params.numStates;

         if nOdors==2
          % neurons for odor 2
         all_neurons(s+1:s+params.numStates,n+1:n+params.numStates) = ones(params.numStates,params.numStates);
         end
         x = all_neurons;


    end


   % Q values for the states of the agents 
    totalQ = zeros(size(x,2),params.numActions);

%% Run the experiment : 
for i=1:numReps

     % Random start location - if start position is [1 100] odor 1 is present ,
     % otherwise odor 2 is present. So each odor is presented with probability 0.5

     params.start = randi(numSessions,1,1);

     % Build the grid world enviroment
    [GW, Q, states, transitionList] = buildGridEgo(params);
    

    % deifne the weight matrix
    w = zeros(size(x, 1), params.numActions);

    % run the td-learning algorith
    [Q1, numSteps1, w1,output] = QLearningFuncApproxEgo(GW, Q, params, states, transitionList, w, x);

    % save some output
    numStepsCon1(i, :) = numSteps1;
    totalQ = totalQ+Q1;
end


% find the maximum valued action in each state:
for j=1:length(totalQ)
     [m,ind]= max(totalQ(j,:));
        % Choose max Q value
     maxQ(j) = m;
end
 

%% Plot results

figure(1);
imagesc(x)
ylabel('states')
xlabel('input layer neurons')
title('input layer representation')


figure;
plot(mean(numStepsCon1,1))
title('Number of Steps to Reward')
xlabel('Session')
ylabel('Number of Steps')

