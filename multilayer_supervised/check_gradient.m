function check_gradient(theta, ei, data, labels, grad, layer_indexes, bias_indexes)

	fprintf('==== check gradient\n');

	for l=1:size(layer_indexes) 

		% we compute two ranges :
		% range1 a range for the random selection of parameters, excluding bias
		% range2 a range for the random selection of biases
		% so that we test both type of calculus
		if l>1
			range1 =  bias_indexes(l) - layer_indexes(l-1);
			base_params = layer_indexes(l-1)-1;
			base_biases = bias_indexes(l)-1;
		else 
			range1 = bias_indexes(l)-1;
			base_params = 0;
			base_biases = 0;
		end%if;	
		range2 = layer_indexes(l) - bias_indexes(l);

		fprintf('layer %d size %d + %d = %d\n',l,range1,range2,range1+range2);

		if  (ei.layer_type(l))~= 3 % not pooling
			check_gradient_inrange(l,range1,'params',base_params,theta, ei, data, labels, grad);		
			check_gradient_inrange(l,range2,'bias',  base_biases,theta, ei, data, labels, grad);
		else	
			fprintf('no checks on pooling layer\n');
		end%if

	end%for

end%function