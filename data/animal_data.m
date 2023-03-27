

% Code to extract animal behavior from Olivia M. data:
rawData = readtable('E02d49.csv');
% load without header
data = readtable('E02d49.csv', 'ReadVariableNames', false);

% turn into matrix
data = data{:,:};

% define parameters:
likelihood_threshold = 0.85; 
cols = 5;
rows = 5;
min_pos = 80;
max_pos = 500;

% visualize raw data 
hold on
scatter(data(data(:,4)>likelihood_threshold,2),data(data(:,4)>likelihood_threshold,3))
scatter(data(data(:,7)>likelihood_threshold,5),data(data(:,7)>likelihood_threshold,6))
legend('Nose','Head')

% Discreteize position into 5 bins ( 5 by 5 grid world)
egdes = min_pos:(max_pos - min_pos)/5:max_pos;

% x position into rows
data_head_y = data(data(:,7)>likelihood_threshold,5);
data_head_row = discretize(data_head_x,egdes);

% y position into columns
data_head_x = data(data(:,7)>likelihood_threshold,6);
data_head_col = discretize(data_head_y,egdes);

% row and col
data_head = [data_head_row , data_head_col];

% Get rid of nans aka mice outside of arena:
data_head = data_head(and(~isnan(data_head(:,1)),~isnan(data_head(:,2))),:);

data_edit = diff(data_head);
data_edit = [ones(1,2); data_edit];

data_head_mod = data_head(or(data_edit(:,1)~=0,data_edit(:,2)~=0),:);


% end of one epside:
% reach south-west port
SW_port = and(data_head_mod(:,1) == 1,data_head_mod(:,2) == 1);
% reach north-east port
NE_port = and(data_head_mod(:,1) == 5,data_head_mod(:,2) == 5);

grid_world = reshape(1:25,5,5);

for i=1:length(data_head_mod)
    state(i) = grid_world(data_head_mod(i,1),data_head_mod(i,2));
    if state(i) == 1 
        grid_world = grid_world + 50;
    elseif state(i) == 25
       grid_world = grid_world + 75; 
    elseif state(i) == 5 || state(i) == 21
        grid_world = reshape(1:25,5,5);
    end
end

hist(state,100)

movies

