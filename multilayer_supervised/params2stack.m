function stack = params2stack(params, ei)

% Converts a flattened parameter vector into a nice "stack" structure 
% for us to work with. This is useful when you're building multilayer
% networks.
%
% stack = params2stack(params, netconfig)
%
% params - flattened parameter vector
% ei - auxiliary variable containing 
%             the configuration of the network
%


% Map the params (a vector into a stack of weights)
depth = numel(ei.layer_sizes);
stack = cell(depth,1);
% the size of the previous layer
prev_size = ei.input_dim; 
% mark current position in parameter vector
cur_pos = 1;

for d = 1:depth
    % Create layer d
    stack{d} = struct;

    switch (ei.layer_type(d))

    case 1 %connected
        hidden = ei.layer_sizes(d);
        % Extract weights
        wlen = hidden * prev_size;
        stack{d}.W = reshape(params(cur_pos:cur_pos+wlen-1), hidden, prev_size);
        cur_pos = cur_pos+wlen;
        % Extract bias
        blen = hidden;
        stack{d}.b = reshape(params(cur_pos:cur_pos+blen-1), hidden, 1);
        cur_pos = cur_pos+blen;        
        prev_size = hidden;

    case 2 %convolve
        [imageDim,filterDim,convDim,numFilters,poolDim,numFiltersPrec]=get_parameters(ei,d);
        wlen = filterDim * filterDim * numFiltersPrec * numFilters;
        stack{d}.W = reshape(params(cur_pos:cur_pos+wlen-1), filterDim,filterDim,numFiltersPrec,numFilters);
        cur_pos = cur_pos+wlen;
        blen = numFilters;
        stack{d}.b = reshape(params(cur_pos:cur_pos+blen-1), numFilters, 1);
        cur_pos = cur_pos+blen;        
        % Set previous layer size
        prev_size = convDim^2 * numFilters; 

    case 3 % pooling
        % do nothing, there are no parameters at all
        [imageDim,filterDim,convDim,numFilters,poolDim,numFiltersPrec]=get_parameters(ei,d);
        prev_size = imageDim^2 / poolDim^2 * numFilters;

    otherwise
        fprintf('invalid layer type %s\n',char(ei.layer_type(d,:)));
    end%switch

end%for

end