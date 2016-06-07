function [ stack ] = initialize_weights( ei )
%INITIALIZE_WEIGHTS Random weight structures for a network architecture
%   ei describes a network via the fields layerSizes, inputDim, and outputDim 

%% initialize hidden layers stack
stack = cell(1, numel(ei.layer_sizes));

    for layer = 1 : numel(ei.layer_sizes)
        % get layers parameters
        [imageDim,filterDim,convDim,numFilters,poolDim,numFiltersPrec,prev_size,cur_size]=get_parameters(ei,layer);
        
        if prev_size<=0 || cur_size<=0
            error('invalid layer size');
        end%if     


        if    compare_string(ei.activation_functions(layer,:),'tanh    ') ==1
            numerator=6;
        elseif compare_string(ei.activation_functions(layer,:),'sigmoid ') ==1
            numerator=4;
        else
            numerator=6;
        end%if    

        % Xaxier's scaling factor, See: X. Glorot, Y. Bengio. Understanding the difficulty of training deep feedforward neural networks. AISTATS 2010
        scale = sqrt(numerator) / sqrt(prev_size + cur_size) * ei.weight_scaling;

        fprintf('initializing layer %d: ',layer); 

        switch (ei.layer_type(layer))

            case 1 %connected        

                stack{layer}.W = (rand(cur_size, prev_size)*2-1)*scale;
                stack{layer}.b = zeros(cur_size, 1);

                fprintf('connected layer %d\n',layer);
                size(stack{layer}.W)
                size(stack{layer}.b)

            case 2 %convolve        

                if ei.random==true
                    stack{layer}.W = (rand(filterDim,filterDim,numFiltersPrec,numFilters)*2-1)*scale;      
                else    
                    stack{layer}.W =  ones(filterDim,filterDim,numFiltersPrec,numFilters)/10;      
                    fprintf('filters\n');
                    stack{layer}.W 
                end%if
                stack{layer}.b = zeros(numFilters, 1);                    

                fprintf('connected layer %d\n',layer);
                size(stack{layer}.W)
                size(stack{layer}.b)

            case 3 %pooling        

                % nothing to do as there are no paramters
 
                fprintf('nothing to initialize for pooling layer %d\n',layer);
 
            otherwise
                fprintf('invalid layer type %s\n',char(ei.layer_type(layer,:))); 
        end%switch

        %fflush(stdout);
    end

end