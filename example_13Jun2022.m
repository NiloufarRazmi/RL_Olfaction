
numContexts = 1;
numRuns = 100;
params = setParams(numContexts);

funcApprox = true;
meanNorm = true;
params.contexts{1} = [1:params.numStates/numContexts]; % list of states in context 1
goals = zeros(numContexts, 1);
numReps = 1;
totalQ = zeros(100,4);
jointRep = true;
makePlots = true;

for i=1:numReps

     % Start location
     params.start = randi(50,1,1);

     % Build the grid world enviroment
    [GW, Q, states, transitionList] = buildGrid(params);
    
    % Define states as 1)joint representation of location and odor or 2)
    % representation of location alone
    if jointRep
       y = eye(100,100);
       x=y;
    else 
        y = eye(25,25);
        x = repmat(y,4,4);
    end

    w = zeros(size(x, 1), params.numActions);

    context = 1;
    [Q1, numSteps1, w1,state_occupancy1,state_mov1] = QLearningFuncApprox(GW, Q, params, states, transitionList, w, x);
    numStepsCon1(i, :) = numSteps1;
    totalQ = totalQ+Q1;
end


% find the maximum valued action in each state:
for j=1:length(totalQ)
     [m,ind]= max(totalQ(j,:));
        % Choose max Q value
     maxQ(j) = m;
end

if makePlots
    makeAllPlots
end