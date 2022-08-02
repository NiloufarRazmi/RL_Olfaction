
%% Set up the task
numContexts = 1;
numRuns = 100;
params = setParams(numContexts);

funcApprox = true;
%shuffle = 2; % maybe categorical? 2 = orthogonal?
meanNorm = true;


numBasesReps = 1; % must be a square
%mat = [eye(params.numStates./2) eye(params.numStates./2)];

params.contexts{1} = [1:params.numStates/numContexts]; % list of states in context 1
params.contexts{2} = [(params.numStates/numContexts + 1):params.numStates]; % list of states in context 2

% Compare numSteps averaged over numRuns for all three variations of
% shuffling
numStepsCon1 = zeros(numRuns, params.numSessions, 3);
numStepsCon2 = zeros(numRuns, params.numSessions, 3);

aveNumSteps1 = zeros(3, params.numSessions);
aveNumSteps2 = zeros(3, params.numSessions);

allPlots = true;

%% Run the task

for j = 0:2
    shuffle = j; % 0 is no shuffling, 1 is random, 2 is orthogonal
    for run = 1:numRuns

        goals = zeros(numContexts, 1);

        % Randomly generate the goal for each context
        for i = 1:numContexts
            %randomIndex = randi(length(params.contexts{i}), 1);
            
            if i ==1 %this will be choosing the goal not randomly, but
            %rather through user input.
               randomIndex=11;
            else
               
               randomIndex=15;
            end
            
            selectedGoal = params.contexts{i}(randomIndex);
            
            goals(i) = selectedGoal;
        end

        [GW, Q, states, transitionList] = buildGrid(params, goals);

        if funcApprox
            %x = repmat(mat, numBasesReps, 1);
            y = [MakeGridCells(1);MakeGridCells(2);MakeGridCells(3);MakeGridCells(4)];
            y = y(1:29,:);
            z = [MakeRotatedGridCells(1);MakeRotatedGridCells(2);MakeRotatedGridCells(3);MakeRotatedGridCells(4)];
            z = z(1:29,:);
            x = [y,y];
            w = zeros(size(x, 1), params.numActions);
            w = zeros(size(x, 1), params.numActions);


            context = 1;
            [Q1, numSteps1, w1] = QLearningFuncApprox(context, GW, Q, params, states, goals, transitionList, w, x);
            numStepsCon1(run, :, j+1) = numSteps1;
            context = 2;
            if shuffle>=1
                x = shuffleBasis(params, x, numBasesReps, logical(shuffle-1));
            end
            if meanNorm==true
                w1=w1-nanmean(w1);
            end

            [Q2, numSteps2, w2] = QLearningFuncApprox(context, GW, Q1, params, states, goals, transitionList, w1, x);
            numStepsCon2(run, :, j+1) = numSteps2;
        else
            

        end
    end

    aveNumSteps1(j+1, :) = mean(numStepsCon1(:, :, j+1));
    aveNumSteps2(j+1, :) = mean(numStepsCon2(:, :, j+1));
end

makePlots;

% Future work:
% Go back to Context 1
% Add in barriers
% No normalization
% Look at which actions it is taking and which states it is going to, some
% tool to plot
% make function to plot state space and agent moving through environment
% simulations
% get rid of need to normalize
