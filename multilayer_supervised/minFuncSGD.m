function [opttheta] = minFuncSGD(funObj,theta,ei,data,labels,options)
% Runs stochastic gradient descent with momentum to optimize the
% parameters for the given objective.
%
% Parameters:
%  funObj     -  function handle which accepts as input theta,
%                data, labels and returns cost and gradient w.r.t
%                to theta.
%  theta      -  unrolled parameter vector
%  data       -  stores data in m x n x numExamples tensor
%  labels     -  corresponding labels in numExamples x 1 vector
%  options    -  struct to store specific options for optimization
%
% Returns:
%  opttheta   -  optimized parameter vector
%
% Options (* required)
%  epochs*     - number of epochs through data
%  alpha*      - initial learning rate
%  minibatch*  - size of minibatch
%  momentum    - momentum constant, defualts to 0.9


%%======================================================================
%% Setup
assert(all(isfield(options,{'epochs','alpha','minibatch'})),...
        'Some options not defined');
if ~isfield(options,'momentum')
    options.momentum = 0.9;
end;
epochs = options.epochs;
alpha = options.alpha;
minibatch = options.minibatch;
m = length(labels); % training set size
% Setup for momentum
mom = 0.5;
momIncrease = 20;
velocity = zeros(size(theta));

%%======================================================================
%% SGD loop
it = 0;
first_loop=true;

for e = 1:epochs
    
    % randomly permute indices of data for quick minibatch sampling
    rp = randperm(m);
    
    fprintf('EPOCH %d\n',e);

    for s=1:minibatch:(m-minibatch+1)
        it = it + 1;

        % increase momentum after momIncrease iterations
        if it == momIncrease
            mom = options.momentum;
        end;

        % get next randomly selected minibatch
        mb_data = data(:,rp(s:s+minibatch-1));
        mb_labels = labels(rp(s:s+minibatch-1));

        % evaluate the objective function on the next minibatch
        [cost grad] = funObj(theta,ei,mb_data,mb_labels);

        switch (options.algo)

        case 1 % basic
            fprintf('basic sgd\n');
            velocity = mom * velocity + alpha * grad;
            theta = theta - velocity;

        case 2 % adagrad
            fprintf('adagrad sgd\n');
            if first_loop
                grad_sq = grad .* grad;                % initiaize grad_sq using the size and value of gradient
            else
                grad_sq = grad_sq + grad .* grad; 
            end%if;    
            grad_norm = sqrt(grad_sq)+1e-10;                  % the norm of the gradient in the time dimension
            velocity = mom * velocity + alpha * (grad ./ grad_norm);
            theta = theta - velocity;
        end%switch

        if first_loop || (cost < last_cost)
            theta_min = theta;
            iteration_min=it;
            last_cost=cost;
        end%if

        first_loop=false;
        
        fprintf('Epoch %d, alpha = %d : Cost on iteration %d is %f, best cost %d\n',e,alpha,it,cost,last_cost);
    end;

    % aneal learning rate by factor after each epoch
    alpha = alpha/options.annealing;

end;

if last_cost>0
    opttheta = theta_min;
    fprintf('using best cost %d for iteration %d\n',last_cost,iteration_min);
else    
    opttheta = theta;
end%if

fprintf('SGD end\n');

end
