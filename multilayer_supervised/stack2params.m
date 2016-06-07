function [params,layer_indexes,bias_indexes] = stack2params(stack,ei)

% Converts a "stack" structure into a flattened parameter vector and also
% stores the network configuration. This is useful when working with
% optimization toolboxes such as minFunc.
%
% [params, layer_indexes] = stack2params(stack)
%
% layer_indexes = index of layers into params vector for test purposes
%
% stack - the stack structure, where stack{1}.w = weights of first layer
%                                    stack{1}.b = biases  of first layer
%                                    stack{2}.w = weights of second layer
%                                    stack{2}.b = biases  of second layer
%                                    ... etc.

params = [];
layer_indexes=[];%1
bias_indexes=[];%-1
    
    for d = 1:numel(stack)

        switch ei.layer_type(d)

        case 1 % connected
            % This can be optimized. But since our stacks are relatively short, it is okay
            params = [params ; stack{d}.W(:) ];
            bias_indexes=[bias_indexes;size(params,1)+1];
            params = [params ; stack{d}.b(:) ];
            layer_indexes=[layer_indexes;size(params,1)+1];
            % Check that stack is of the correct form
            assert(size(stack{d}.W, 1) == size(stack{d}.b, 1), ...
                ['The bias should be a *column* vector of ' ...
                 int2str(size(stack{d}.W, 1)) 'x1']);
            % no layer size constrain with conv nets
            if d<numel(stack) && ei.layer_type(d)==1 && ei.layer_type(d+1)==1  % both connected
                    assert(mod(size(stack{d+1}.W, 2), size(stack{d}.W, 1)) == 0, ...
                        ['The adjacent layers L' int2str(d) ' and L' int2str(d+1) ...
                         ' should have matching sizes.']);
            end%if        

        case 2 % convolved
            % This can be optimized. But since our stacks are relatively short, it is okay
            params = [params ; stack{d}.W(:) ];
            bias_indexes=[bias_indexes;size(params,1)+1];
            params = [params ; stack{d}.b(:) ];
            layer_indexes=[layer_indexes;size(params,1)+1];

        case 3 % pooling 
            bias_indexes=[bias_indexes;size(params,1)+1];
            layer_indexes=[layer_indexes;size(params,1)+1];

        otherwise
            fprintf('invalid layer type %s\n',char(ei.layer_type(l,:))); 
        end%switch


    end



end