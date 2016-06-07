function d=tanh_d(z)
% derivative of tanh as a function of the tanh values
% tanh is natively defined by Matlab

	a=tanh(z);
	d = 1 - a .* a;

end%function