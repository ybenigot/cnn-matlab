function d=sigmoid_d(z)
% derivative of sigmoid 

%	d = a .* (1 - a);% as a function of the sigmoid values
	e = exp(-z);
	d = e ./ ((1 + e) .^ 2);

end%function