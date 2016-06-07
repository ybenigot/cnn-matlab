fprintf('hyperparameters CNN1\n');

%% populate ei with the network architecture to train
% ei is a structure you can use to store hyperparameters of the network
% the architecture specified below should produce  100% training accuracy
% You should be able to try different network architectures by changing ei
% only (no changes to the objective function code)


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

ei.layer_type= [convolve,connected,connected];
ei.pool_types= ['max';'N/A';'N/A'];
% sizes of all hidden layers and the output layer ; number of features for convolutional layer
ei.layer_sizes = [-1, 60, ei.output_dim];

% hyperparameters for convulutional layer(s)
ei.image_size = [ei.input_pixels,0,0];				% pixel size of image for each layer
ei.filter_sizes = [9,0,0];							% filter size for convolutional layers
ei.numFilters = [10,0,0];
ei.pool_dim = [2,0,0];

% scaling parameter for l2 weight regularization penalty
ei.lambda = 0.3;
ei.sgd=0;

% debugging parameters
ei.random=true;
ei.debug_level=0;
ei.check=false;
ei.stop_backprop=false;

%% setup minfunc options
options = [];
options.display = 'iter';
options.maxFunEvals = 1e6;
options.Method = 'lbfgs';
%options.progTol = 1e-4;

% check gradient
ei.check=false;

ratio_of_dataset=10;
[data_train, labels_train, data_test, labels_test] = load_preprocess_mnist(ratio_of_dataset);
