function [cost,delta,scaling] = logreg_cost(ei,y,h,stack,m,lambda)

% cost function based on logistic function hypothesiss


K = ei.output_dim;											% typically 10

l = (y * ones(1,K)) == ones(m,1) * (1:K) ; 					% y size 60000x1, l(y(i)=k) , size: m,K

%% compute cost
cost = - (1/m) * sum((l' .* log(h) + (1-l') .* log (1-h))(:)); 	

% regularization of cost
depth = numel(ei.layer_sizes);
for i=1:depth
	W = stack{i}.W;
	cost = cost + (lambda / (2*m)) * sumsq(W(:));   
end%for

delta=h - l'; 

scaling=1/m; % multiplier for gradient

if ei.debug_level>=3
	printf('hypothesis\n');
	h
	printf('target output:\n');
	y
	printf('delta - same dims as h\n');
	delta
end%if

end


