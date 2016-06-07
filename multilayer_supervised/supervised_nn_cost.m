function [ cost, grad, pred_prob] = supervised_nn_cost( theta, ei, data, labels, pred_only, cost_only, show_images)

global train_stats;

po = false;
if exist('pred_only','var')
  po = pred_only;
end;

co = false;
if exist('cost_only','var')
  co = cost_only;
end;

show = false;
if exist('show_images','var')
  show = show_images;
end;

[pred_prob,stack,hAct,maxAct,netInp] = prediction(theta, ei, data, labels, po, co, show);

if po
  cost = -1; 
  grad = [];  
  return;
end;

y =labels;
m = size(y,1);
lambda = ei.lambda;

costFunc=str2func(strtrim(ei.cost_function));		% cost function is defined by an hyperparameter

[cost, delta, scaling] = costFunc(ei, y, pred_prob, stack, m, lambda);

if co 	% stop after computing cost
  grad = [];  
  return;
end;

fprintf('computed cost: %d\n',cost);
%fflush(stdout);

train_stats=[train_stats;zeros(1,numel(ei.layer_sizes)+1)];	% add a line of zeroes
train_stats(end,1)=cost;

grad = backprop(theta, ei, data, labels, delta, scaling, stack, hAct, maxAct, netInp, m, lambda);

clear stack hAct delta_next netInp;

end


