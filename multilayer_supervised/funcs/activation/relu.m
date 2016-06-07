function h=relu(z)

% WARNING : does not work with log cost function because of the possibility of a zero output
% if needed replace zero by epsilon=1e-9

	h = max(0,z);									% size layer_size x m

end%function
