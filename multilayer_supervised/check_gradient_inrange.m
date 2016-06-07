function check_gradient_inrange(l,range,range_name,base_index,theta, ei, data, labels, grad)

	p=size(theta,1);
	epsilon=1e-6; % typical 1e-6 or 1e-4

	%randomly select indexes
	nbsamples=min(100,range);	% number of samples per layer

	if (nbsamples==range)
		perm = 1:range;
	else
		perm = randperm(range,nbsamples);
	end%if	

	difference=zeros(nbsamples,1);
	j=1;

	ok=0;
	unprecise=0;
	ko=0;

	for i=perm

		grad_index=i+base_index;

		est_grad = estimate_derivative(theta,ei,data,labels,(1:p==grad_index)',epsilon);

		computed_grad = grad(grad_index);

		if est_grad==0
			if computed_grad==0
				difference(j)=0;
			else
				difference(j) = (computed_grad-est_grad) / computed_grad;
			end%if				
		else	
			difference(j) = (computed_grad-est_grad) / est_grad;
		end%if

		gradient_error = 2*(est_grad - computed_grad)/(abs(est_grad) + abs(computed_grad)+ 1e-9);% never divide by zero

		if computed_grad~=0
			ratio_grad = est_grad/computed_grad;
		else
			ratio_grad = 0;
		end%if;	

		if abs(gradient_error) >= 1e-1
			%fprintf('layer %d # %2d (%4d) for %s gradient ERROR : est. %f comp. %f ratio %f \n',l,j,grad_index,range_name,est_grad,computed_grad,ratio_grad);
	        ko=ko+1;		
		elseif abs(gradient_error) >= 1e-3
			%fprintf('layer %d # %2d (%4d) for %s gradient unprecise : est. %f comp. %f ratio %f \n',l,j,grad_index,range_name,est_grad,computed_grad,ratio_grad);
			unprecise=unprecise+1;
		else
			%fprintf('layer %d # %2d (%4d) for %s gradient OK : est. %f comp. %f ratio %f \n',l,j,grad_index,range_name,est_grad,computed_grad,ratio_grad);
			ok=ok+1;
		end%if	

		j = j + 1;

	end%for	

	fprintf('LAYER %d OK %d, unprecise %d, KO %d \n',l,ok,unprecise,ko);

	%fflush(stdout);

end%function