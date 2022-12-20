%% Backprop to RL
% 



% 1) inputs need to reflect position in arena and odor (NOT CONJUNCTIONS)
% 2) outputs need to reflect action values
% 3) actions are selected via softmax on output neuron activity. 
% 4) RPE requires knowing value of new state -- so this will require a
%     forward pass using "new state" inputs. 








% Basically, i'm just coding up what is here:
% https://en.wikipedia.org/wiki/Backpropagation
% under heading: Finding the derivative of the error

% Goal is to make a simple multi layer network, feed it an "exclusive or"
% problem, then use backprop to solve it.

% Originally I was going to code up a 3 layer network for simplicity -- but
% it appears to work better on x/or if i give it more layers. I think its
% because the non-linearity (logistic) is not very extreme, and a couple
% layers can work together to make it a sharper non-linearity. 

% Currently uses a logistic activation function:
% f(x) = 1./(1+exp(-x))

% Here is what it looks like:
% x=-5:.1:5
% y=1./(1+exp(-x))
% plot(x, y)


clear classes


% Main options:
nLayers=5; % set number of layers for neural network
problem = 2; % 3 = super hard, 2 = hard, 1 = easy. See below for actual functions. 


%% Step 1: generate data
nTrain=1000000;
nTest =100;

nTot=nTrain+nTest;
% create two normal input channels
X = mvnrnd([0,0],[10, 10], nTot);
%Y is positive if X(1) & X(2) are positive, OR if X(1) and X(2) are negative. 


switch problem
    case 1
        % Easy problem: LINEAR 
        Y = sum(X,2)>0;
    case 2
        % Hard problem = EXCLUSIVE OR
        Y = sign(X(:,1).*X(:,2))./2  +.5;
    case 3
        % Super hard problem = CIRCLE
        Y = sqrt(X(:,1).^2+X(:,2).^2)<4;   
end


hold on
plot([-2, 2], [0, 0], '--k')
plot([0, 0],[-2, 2],  '--k')
plot(X(Y==1,1), X(Y==1,2), 'o', 'markerFaceColor', 'r', 'lineWidth', 1, 'markerEdgeColor', 'k')
plot(X(Y==0,1), X(Y==0,2), 'o', 'markerFaceColor', 'b', 'lineWidth', 1, 'markerEdgeColor', 'k')
ylabel('Feature 1')
xlabel('Feature 2')

saveas(gcf, 'X_or_nonlinearSepExample.eps', 'epsc2')
close all

%% Step 2: Build network

% determine number of units in each layer
nOutputUnits=1;
nInputUnits=size(X,2);
nHiddenUnits=10;
initVar=1;
nonLin=[false,  true(1, nLayers-2), true];

% Create initial weight matrices
nUnits=[nInputUnits, repmat(nHiddenUnits, 1, nLayers-2), nOutputUnits];
clear wtMatrix
for i = 1:(nLayers-1)
    wtMatrix{i}=normrnd(0, initVar, nUnits(i), nUnits(i+1));
end


%wtMatrix{1}=[10, 10, -10, -10; 10, -10, 10, -10];


%% Step 3: Train network

LR=.001;

    allError=nan(nTot,1);
    catPredict=nan(nTot,1);

for i = 1:nTrain

    % Generate model prediction (forward pass of activity through units):
    activity=cell(nLayers,1); 
    for j = 1:nLayers
        % Determine layer input:
        if j ==1
             input=X(i,:); % THIS WILL BE YOUR POSITION/ODOR!!!!!
        else
             input=activity{j-1}*wtMatrix{j-1};
        end
        
        % Apply non-linearity
        if nonLin(j)        
        activity{j}=  1./(1+exp(-input)); %
        else
        activity{j}=input;
        end

    end
    
    
    % Take an action! softmax over actions or similar
    
    
    % incorporate your model of the task, to dedtermine where agent
    % actually goes. 
    
    % Now you need to do another forward pass, to see how good the new
    % state is so that you can compute the RPE below. 
    
    
    % your RPE will differ from the one below, should look something like
    % this:
    %RPE =  R - X(S)*W+ DISCOUNT*max(X(S')*W)
   
    
    % Backpropagate errors to compute gradients for all layers:    
    for j = nLayers:-1:1
        % Determine layer input:
        if j==nLayers
             % IF there is nonlinearity, should multiply by derivative of 
             % activation with respect to input (activity.*(1-activity)) here.  
            delta{j}= (Y(i)-activity{j}).* (activity{j}.*(1-activity{j}))';   % THIS SHOULD BE REPLACED WITH YOUR RPE! 
            
            % doing this in RL framework means that you'll need one RPE for
            % each output neuron -- so RPE computed above should be
            % associated with the action agent took... all other RPEs
            % should be zero. 
            
        else
             % OK, here is the magic -- neurons in layer j share the
             % gradient (ie. prediction errors) from the next layer
             % according to their responsibility... that is to say, if I
             % project to a unit in next layer with a strong weight, 
             % then i inherit the gradient (PE) of that unit. 
             
             %        
             delta{j}= wtMatrix{j}* delta{j+1} .* (activity{j}.*(1-activity{j}))';
        end
    end

    % Update weight matrices according to gradients and activities:
    for j = 1:length(wtMatrix)
        wtMatrix{j}=wtMatrix{j}+LR.*activity{j}'*delta{j+1}';
    end

    %store error:
    allError(i)=delta{end};
    catPredict(i)=activity{end}>.5;
end

close all


Bins=round(linspace(1, length(allError)));

for i = 1:(length(Bins)-1)
meanError(i)=mean(abs(allError(Bins(i):Bins(i+1))));
end

figure(1)
plot(meanError)
ylabel('Error')
xlabel('Batches')



%% Step 4: Test Network

for i = (nTrain+1):nTot
    % Generate model prediction (forward pass of activity through units):
    activity=cell(nLayers,1); 
    for j = 1:nLayers
        % Determine layer input:
        if j ==1
             input=X(i,:); % initial layer is activated according to input
        else
             input=activity{j-1}*wtMatrix{j-1};
        end
        
        % Apply non-linearity
        if nonLin(j)        
        activity{j}=  1./(1+exp(-input));
        else
        activity{j}=input;
        end

    end
   %store error:
    allError(i)=delta{end};
    catPredict(i)=activity{end}>.5;
end

isTest=false(nTot,1);
isTest(nTrain+1:end)=true;



%% How does train model do on new data? Lets take a look...

close all

figure(2)
subplot(1, 2, 1)
title('Ground Truth')
hold on
% plot([-2, 2], [0, 0], '--k')
% plot([0, 0],[-2, 2],  '--k')
plot(X(Y==1&isTest,1), X(Y==1&isTest,2), 'o', 'markerFaceColor', 'r', 'lineWidth', 1, 'markerEdgeColor', 'k', 'markerSize', 14)
plot(X(Y==0&isTest,1), X(Y==0&isTest,2), 'o', 'markerFaceColor', 'b', 'lineWidth', 1, 'markerEdgeColor', 'k', 'markerSize', 14)
ylabel('Feature 1')
xlabel('Feature 2')


subplot(1, 2, 2)
title('Model Classification')
hold on
% plot([-2, 2], [0, 0], '--k')
% plot([0, 0],[-2, 2],  '--k')
plot(X(catPredict==1&isTest,1), X(catPredict==1&isTest,2), 'o', 'markerFaceColor', 'r', 'lineWidth', 1, 'markerEdgeColor', 'k',  'markerSize', 14)
plot(X(catPredict==0&isTest,1), X(catPredict==0&isTest,2), 'o', 'markerFaceColor', 'b', 'lineWidth', 1, 'markerEdgeColor', 'k', 'markerSize', 14)
ylabel('Feature 1')
xlabel('Feature 2')

saveas(gcf, 'multiLayerPerceptron_ex2.eps', 'epsc2')




