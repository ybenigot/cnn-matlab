function displayParameters(params, ei, layer)


	[imageDim,filterDim,convDim,numFilters,poolDim,numFiltersPrec]=get_parameters(ei,layer);
	wlen = filterDim * filterDim * numFiltersPrec * numFilters;

	        
	weights = reshape(params(1:wlen),filterDim*filterDim,numFiltersPrec*numFilters);
	fprintf('for layer %d ',layer);
	displayData(weights',ei.filter_sizes(1),'displaying parameters...\n',0);


end%function