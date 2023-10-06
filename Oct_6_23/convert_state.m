
function[input] = convert_state(nPlace_states,nOdors,nLights,heading)

% manual hard-coded way for creating input layer activity corresponding to
% each state of the enviroment- future step would include making this better
%  Jan 8 2023

% nPlace_states:
% number of position-related states.

% place cells:
place_input = eye(nPlace_states,nPlace_states);
place_input = repmat(place_input,1,4);

% initialize "input layer" with seperate place, odor and light cells:
input = zeros(nPlace_states + nOdors+nLights,nPlace_states*(nLights+nOdors));

% the first "nPlace_states" neurons are represeting location:
input(1:nPlace_states,:) = place_input;

% loop through 4 possible place-context associations and creat input layer
% activity for odor and light neurons:

for i=1:4
    input(nPlace_states+i,(i-1)*nPlace_states+1:i*nPlace_states) = ones(1,nPlace_states);
end
