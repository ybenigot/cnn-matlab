function cost=test_squared_cost

%% some parameters
% dimension of input features
ei.input_pixels = 28; 
ei.input_dim = ei.input_pixels * ei.input_pixels; %784
% number of output classes
ei.output_dim = 10;
% which type of activation function to use in hidden layers
% feel free to implement support for only the logistic sigmoid function
% this hyper parameter shall have same size as layer_sizes 
% using 'sigmoid' implies that two functions are defined : sigmoid, and its derivative sigmoid_d
ei.activation_functions = ['sigmoid';'sigmoid';'softmax'];
ei.cost_function = 'cross_entropy_cost'; % possibilities are logreg_cost and cross_entropy_cost 

ei.layer_type= [0,0,0];
ei.pool_types= ['max';'max';'N/A'];
% sizes of all hidden layers and the output layer ; number of features for convolutional layer
ei.layer_sizes = [-1, -1, ei.output_dim]; % -1 means N/A because this is a convolution layer

% hyperparameters for convulutional layer(s)
ei.image_size = [ei.input_pixels,10,0];				% pixel size of image for each layer
ei.filter_sizes = [9,9,0];							% filter size for convolutional layers
ei.numFilters   = [30,50,0];
ei.pool_dim     = [2,2,0];

% scaling parameter for l2 weight regularization penalty
ei.lambda = 0.0;
ei.sgd=1;

% debugging parameters
ei.random=true;
ei.debug_level=0;
ei.check=false;
ei.stop_backprop=false;

addpath funcs/cost;

lambda=0;
stack={};
m=2;
y=[2;1];
h=[0 1 0 0 0 0 0 0 0 0 ; 0 1 0 0 0 0 0 0 0 0]';

%% data
[cost,delta,scaling] = squared_cost(ei,y,h,stack,m,lambda)


end%function