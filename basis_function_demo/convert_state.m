
function[input] = convert_state(n_place,n_odors,n_ports)



% place cells:
place_input = eye(n_place,n_place);
place_input = repmat(place_input,1,(n_ports+n_odors));

% initialize "input layer" with seperate place, odor and light cells:
input = zeros(n_place + (n_odors+n_ports),n_place*(n_odors+n_ports));

% the first "nPlace_states" neurons are represeting location:
input(1:n_place,:) = place_input;

% loop through 4 possible place-context associations and creat input layer
% activity for odor and light neurons:

for i=1:n_odors+n_ports
    input(n_place+i,(i-1)*n_place+1:(i)*n_place) = ones(1,n_place);
end
