function [cost,delta,scaling] = cross_entropy_cost(ei,y,h,stack,m,lambda)

%% compute cost

%printf('cost computation\n');

% h = pred_prob
K = ei.output_dim;											% typically 10

l = (y * ones(1,K)) == ones(m,1) * (1:K) ; 					% y size 60000x1, l(y(i)=k) , size: m,K

l2 = (l' .* log(h));

cost = - sum(l2(:)); 	   						% cross entropy
%regularization of cost - same as for squared error cost
if lambda > 0
	depth = numel(ei.layer_sizes);
	for i=1:depth
		if (ei.layer_type(i))~= 3 % not pooling
			W = stack{i}.W;
			cost = cost + (lambda / (2*m)) * sumsq(W(:));   
		end%if	
	end%for
end%if

delta=h - l'; 

scaling=1; % multiplier for gradient

end



