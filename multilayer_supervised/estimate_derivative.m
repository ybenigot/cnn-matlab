function derivate=estimate_derivative(theta,ei,data,labels,delta_direction,epsilon) 

% estimate the partial derivative of a function given the current point in space, theta,
% the direction of variation delta_direction,
% and the epsilon to use
% using the weel known formula :
%  F'(x) = ( F(x+epsilon) - F(x-epsilon) )/ (2*epsilon) as an approximation of the derivative's value
%

	delta_vector = epsilon * delta_direction;
		
	thetaplus  = theta + delta_vector;
	thetaminus = theta - delta_vector;
		
	costplus  = supervised_nn_cost( thetaplus,  ei, data, labels, false, true);
	costminus = supervised_nn_cost( thetaminus, ei, data, labels, false, true);
		
	derivate = (costplus - costminus) / (2*epsilon);

end%function