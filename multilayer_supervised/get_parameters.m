function [imageDim,filterDim,convDim,numFilters,poolDim,numFilterPrec,prev_size,cur_size,dropout] = get_parameters(ei,layer)

	imageDim = ei.image_size(layer);
    filterDim = ei.filter_sizes(layer);
	convDim = get_parameter_convDim(ei,layer);
	numFilters = ei.numFilters(layer);
	poolDim = ei.pool_dim(layer);
    cur_size = ei.layer_sizes(layer);
    dropout=ei.dropout(layer);

	if layer > 1
		numFilterPrec = ei.numFilters(layer-1);
        switch (ei.layer_type(layer-1))
        case 1 % connected
            prev_size = ei.layer_sizes(layer-1);
        case 2 % convolve
            convDimPrec = get_parameter_convDim(ei,layer-1);
            prev_size = convDimPrec^2 * numFilterPrec;
        case 3 % pooling
            poolDimPrec = ei.pool_dim(layer-1);
            prev_size = ei.image_size(layer-1)^2 / poolDimPrec^2  * numFilterPrec ;
        end%switch
    else
        prev_size = ei.input_dim;
		numFilterPrec = ei.input_depth; 
    end;

    if prev_size==0 || cur_size==0
        fprintf('layer : %d type of previous layer %d\n',layer,ei.layer_type(layer-1));
        error('prev size or cur size 0');
    end%if    
		
end%function		