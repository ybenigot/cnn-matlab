function convDim=get_parameter_convDim(ei,layer)

	imageDim = ei.image_size(layer);
    filterDim = ei.filter_sizes(layer);
	convDim = imageDim - filterDim + 1;

end%function