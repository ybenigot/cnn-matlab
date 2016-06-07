function [pred_prob,stack,hAct,maxAct,netInp] = prediction(theta, ei, data, labels, pred_only, cost_only, show)

start=cputime;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
if not(pred_only)
	hAct = cell(numHidden+1, 1);		% stores activation
	maxAct =  cell(numHidden+1, 1);		% stores coordinates of selected pixel after max pooling
	netInp =  cell(numHidden+1, 1);		% net input to derivation function, memorized for backprop
else 
	hAct=[];	
	maxAct=[];	
	netInp=[];		% stores coordinates of selected pixel after max pooling
end%if

%% forward propagation

%fprintf('>>>> forward propagation\n');

depth = numel(ei.layer_sizes);
X = data;		% size n x m
m = size(X,2);	
numImages = m;

for layer=1:depth

	[imageDim,filterDim,convDim,numFilters,poolDim,numFiltersPrec,cur_size,prev_size,dropout]=get_parameters(ei,layer);

   	switch (ei.layer_type(layer))

    case 1 %connected
    	%fprintf('connected layer #%d\n',layer);
		W = stack{layer}.W;											% size : next_layer x layer
		b = stack{layer}.b;
		g = str2func(strtrim(ei.activation_functions(layer,:)));	
 		z = (W * X) + (b * ones(1,m)) ;							% size layer_size x m	
		h = g(z);
		netInp{layer} = z;		 					% records for back propagation

    case 2 %convolve
		W = stack{layer}.W;											% size : next_layer x layer
		b = stack{layer}.b;
		g = str2func(strtrim(ei.activation_functions(layer,:)));

		%fprintf('convolutional layer #%d numFilter = %d\n',layer,numFilters);
		
		images=reshape(X, imageDim, imageDim, numFiltersPrec, numImages);

		% this will be a sum over all featureMaps in the former layer
		convolvedFeatures = zeros(convDim, convDim, numImages,numFilters); % filter must be the last coordinate in the lopp
		 netInputFeatures = zeros(convDim, convDim, numImages,numFilters);
		
		% convolution and summation on all feature maps from previous layer
		for filterNum = 1:numFilters
			for filterPrecNum = 1:numFiltersPrec
				% W is a 4D tensor depending on input feature map and output feature map 
				% we convert that in a 2D tensor, the filter, and repeat it for all images, creating a 3D Tensor
				% FIXME that tensor can be very big, maybe put the image dimension as the first dim and use broadcasting
				filterRepeated =  reshape(repmat(W(:,:,filterPrecNum,filterNum),1,numImages),filterDim,filterDim,numImages);
				convolvedImages = convolve3(images(:,:,filterPrecNum,:),filterRepeated);
				% sum results of all incoming maps in the output map
				convolvedFeatures(:,:,:,filterNum) = convolvedFeatures(:,:,:,filterNum)+convolvedImages;	
			end%for
			% add bias to the convolution
			convolvedFeatures(:,:,:,filterNum) = convolvedFeatures(:,:,:,filterNum) + b(filterNum,1) ;	
			netInputFeatures(:,:,:,filterNum) = convolvedFeatures(:,:,:,filterNum); 
			% apply nonlinearity per output feature map on all the summed convolutions of input maps and biases
			convolvedFeatures(:,:,filterNum,:) = g(convolvedFeatures(:,:,filterNum,:));	
		end%for

		convolvedFeatures=permute(convolvedFeatures,[1 2 4 3]); % now imageNum is the last coordinate
		netInputFeatures=permute(netInputFeatures,[1 2 4 3]); 

		netInp{layer} = reshape(netInputFeatures,convDim*convDim*numFilters,numImages);

		h = reshape(convolvedFeatures,convDim * convDim * numFilters, numImages);

		% implement dropout if this is training
		if not(pred_only) && (dropout>0)
			r = rand(size(h))>dropout;
			%fprintf('for %d count of 1 = %d, of 0= %d\n',dropout,sum(r(:)==1),sum(r(:)==0));
			h = h .* r * (1/(1-dropout));
        end%if

		if ei.debug_level>=3
			dump('images,convolvedFeatures,h',images,convolvedFeatures,h);
		end%if	

	case 3 % pooling

		%fprintf('pooling layer #%d numFilter = %d\n',layer,numFilters);
		
		images=reshape(X, imageDim, imageDim, numFiltersPrec, numImages);

		pool4 = str2func(strcat(ei.pool_types(layer,:),'_pool4'));
		[h ixFeatures] = pool4(images,poolDim);
		h = reshape(h,imageDim^2 / poolDim^2 * numFiltersPrec, numImages);
		netInp{layer} = h;

		if not(pred_only)
			maxAct{layer} = ixFeatures;		 					% records for back propagation
		end%if

		if ei.debug_level>=3
			dump('images, pool4',images,pool4);
		end%if	



	otherwise
		fprintf('invalid layer type %s\n',char(ei.layer_type(layer,:)));
	end%switch

	X = h;												% init input of next layer

	if not(pred_only)
		hAct{layer} = h;		 							% size layer_size x m 
															% records activations for back propagation
	end%if

	if layer==1 && show==true && ei.layer_type(layer)==2 %convolved
		images=reshape(h,(convDim/poolDim)*(convDim/poolDim),numFilters,numImages);
	%	displayData(images(:,1,1:min(100,m)),convDim/poolDim,'displaying first layer filter 1 as images...\n');
	end%if	
	%fprintf('.');
	%fflush(stdout);

end%for

pred_prob = h;										% size output_layer x m   pred_prob is output

%fprintf('\nend forward propagation after %d s\n',cputime-start);
%fprintf('\nend forward propagation after %d s, prediction for image 1 is :\n',cputime-start);
%PR=pred_prob(:,1)
%fflush(stdout);

clear h z X; 

end