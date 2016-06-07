function [grad] = backprop( theta, ei, data, labels, delta_next, scaling, stack, hAct, maxAct, netInp, m, lambda)

% typical sizes : layer 1 = n = 784, layer 2 = 256 layer 3 = 10
% depth=2
% a1=X
% a2=g(theta1'*a1) -- depth=1
% a3=g(theta2*a2)  -- depth=2
% g'(z)=g(z)(1-g(z))  -- logistic function derivative
% delta(l) = activation - prediction l=1,2 delta is not defined for layter0=input
% delta(2)=theta2' * delta(3) .* g'(z2)
% delta_accumulated =delta_acc(l) = delta(l+1) * a(l)' 
% grad(l)=delta_acc(l)/m

% last layer delta size is 9 should be 10
% decide whether to transpose when coputing gradStack
% layer size shoud match at the end check

if ei.verbose
	fprintf('\n<<<< back propagation\n');
end%if

global train_stats;
start=cputime;

numHidden = numel(ei.layer_sizes) - 1;
depth = numel(ei.layer_sizes);
gradStack = cell(numHidden+1, 1);

%% compute gradients using backpropagation
for layer=depth:-1:1											% layer 0 would be input layer

	if layer>1
		activation_pred = [ones(1,m);hAct{layer-1}];
	else
		activation_pred = [ones(1,m);data];
	end%if
	if layer>1
		derivate=str2func(strcat(strtrim(ei.activation_functions(layer-1,:)),'_d'));	% by convention the derivative has the name of the function

	else
		clear derivate;
	end%if	
				

	switch (ei.layer_type(layer))	
    case 1 %connected			     ----- regular NN backprop for fully connected layer ------

		if ei.verbose
			fprintf('layer %d size %d\n',layer,ei.layer_sizes(layer)); %fflush(stdout);
		end%if

		b = stack{layer}.b;											% size(layer) x 1				ex: 10x1
		W = stack{layer}.W;											% size(layer) x size(layer-1) 	ex:10x25å

		if layer>1
			derivative = [zeros(1,m);derivate(netInp{layer-1})];
			delta = ([b W]' * delta_next) .* derivative;	% delta(layer-1)=delta(2)=theta2T*delta3*g'(z2) - 
																% 1+size(layer-1) x m
																% followed by _d, here we use the derivative for the previous layer
		end%if

		delta_acc = delta_next * activation_pred';				% size(layer)xm*mxsize(layer-1)+1->
																% size(layer)xsize(layer-1)+1	ex:10x6000*6000x257=10x257

		delta_m=(delta_acc / m);								% size(layer)x1+size(layer-1)  ex : 10x257

		%compute gradient for parameters between layer-1 and layer
		gradStack{layer}.W = scaling*m*delta_m(:,2:end) + scaling*lambda*W;		% regularize, except for bias
		gradStack{layer}.b = scaling*m*delta_m(:,1);

		% prepare for next iteration
		delta_next = delta(2:end,:);

	case 2 %convolve	            ----- convolutional upsampling & backprop -----

		b = stack{layer}.b;											% size(layer) x 1				ex: 10x1
		W = stack{layer}.W;											% size(layer) x size(layer-1) 	ex:10x25å

	    [imageDim,filterDim,convDim,numFilters,poolDim,numFiltersPrec,prev_size,cur_size]=get_parameters(ei,layer);

		if ei.verbose
			fprintf('layer %d numFilters %d numFilterPrec %d\n',layer,numFilters,numFiltersPrec); %fflush(stdout);
		end%if

		% convolve the error values after upsampling with the input to get the filter variation
		images = reshape(activation_pred(2:end,:),imageDim,imageDim,numFiltersPrec,m); % remove 1 for bias	

		% error on activation for this layer, as a matrix of images
		delta_next_img = reshape(delta_next,convDim,convDim,numFilters,m);

		gradParam  = zeros(filterDim,filterDim,numFiltersPrec,numFilters);
		gradBias  =  zeros(numFilters,1);

		% compute gradient for parameters and bias : a(n) *+ delta(n+1)
		for filterNum = 1:numFilters  % FIXME : can we convolve on 4D ?
			for filterPrecNum = 1:numFiltersPrec  % FIXME : can we convolve on 4D ?
				product = conv3(squeeze(images(:,:,filterPrecNum,:)),...
					                squeeze(delta_next_img(:,:,filterNum,:)));
				gradParam(:,:,filterPrecNum,filterNum) = sum(product,3);
				gradBias(filterNum,1)= sum(product(:));
			end%for
		end%for

		gradStack{layer}.W = scaling * gradParam; 
		gradStack{layer}.b = scaling * gradBias;

		if layer>1
			% compute derivative for convolved layer 
			act_derivative = derivate(netInp{layer-1});			% g'(layer-1) - convDim^2 as a function of the activation
			act_derivative = reshape(act_derivative,imageDim,imageDim,numFiltersPrec,m);
			delta_next = zeros(imageDim,imageDim,numFiltersPrec,m);

			% compute delta for next layer (delta_up(n+1) *- theta(n)) .* derivative(n)
			for filterPrecNum = 1:numFiltersPrec 
				for filterNum = 1:numFilters  
					delta1=squeeze(delta_next_img(:,:,filterNum,:));
					filter1 = rot90(W(:,:,filterPrecNum,filterNum),2);
					filters=reshape(repmat(filter1,1,m),filterDim,filterDim,m);
					product = conv3full(delta1,filters);
					delta2 = product .* squeeze(act_derivative(:,:,filterPrecNum,:));
					delta_next(:,:,filterPrecNum,:) = delta_next(:,:,filterPrecNum,:) + reshape(delta2,imageDim,imageDim,1,m);
				end%for	
			end%for	
		end%if

		if ei.debug_level>=3
			dump('delta_next, derivative, images, gradient',delta_next_img,delta_next,derivative,images,gradStack{layer});
		end%if	

	case 3 % pooling

	    [imageDim,filterDim,convDim,numFilters,poolDim,numFiltersPrec,prev_size,cur_size]=get_parameters(ei,layer);

		if ei.verbose
			fprintf('layer %d numFilters %d numFilterPrec %d\n',layer,numFilters,numFiltersPrec); %fflush(stdout);
		end%if

		delta_next_img = reshape(delta_next,imageDim/poolDim,imageDim/poolDim,numFilters,m);

		% select the functions according to layer parameters
		upsample4 = str2func(strcat(strtrim(ei.pool_types(layer,:)),'_upsample4')); % upsampling function for reverse of pooling
		% upsample the error values
		delta_next = upsample4(delta_next_img,maxAct{layer},poolDim);

		if ei.debug_level>=3
			dump('delta_next, derivative, images, gradient',delta_next_img,delta_next,derivative,images,gradStack{layer});
		end%if	

	otherwise
		fprintf('invalid layer type %s\n',char(ei.layer_type(layer,:)));
	end%switch

	if ei.layer_type(layer)~=3
		train_stats(end,layer+1)=median(abs([gradStack{layer}.W(:);gradStack{layer}.b(:)]));
	end%if

	clear activation_pred derivative delta_acc grad_acc delta_m W b;



end%for

%fprintf('+\n');
%fflush(stdout);

clear delta,delta_next;

%% reshape gradients into vector
[grad,layer_indexes,bias_indexes] = stack2params(gradStack,ei);

%%for testing only
if ei.check==true
	check_gradient(theta, ei, data, labels,grad,layer_indexes,bias_indexes);
end%if

if ei.verbose
	fprintf('end backward propagation after %d s\n',cputime-start);
end%if
%fflush(stdout);

ei.check=false; % stop checking at the end of backprop
if ei.stop_backprop==true
  error('ITERATION STOPS FOR TEST');
end%if

end


