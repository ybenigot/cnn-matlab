fprintf('hyperparameters TEST MIN 5\n');

%% populate ei with the network architecture to train
% ei is a structure you can use to store hyperparameters of the network
% the architecture specified below should produce  100% training accuracy
% You should be able to try different network architectures by changing ei
% only (no changes to the objective function code)


% dimension of input features
ei.input_pixels = 28; 
ei.input_dim = ei.input_pixels * ei.input_pixels; %784
ei.input_depth=1; % number of input planes 1 for B&W, 3 for RGB

% number of output classes
ei.output_dim = 10;
% which type of activation function to use in hidden layers
% feel free to implement support for only the logistic sigmoid function
% this hyper parameter shall have same size as layer_sizes 
% using 'sigmoid' implies that two functions are defined : sigmoid, and its derivative sigmoid_d
ei.activation_functions = ['relu';'softmax'];
ei.cost_function = 'cross_entropy_cost'; % possibilities are square_cost and cross_entropy_cost 

ei.layer_type= [connected,connected];
% sizes of all hidden layers and the output layer ; number of features for convolutional layer
ei.layer_sizes = [150, ei.output_dim];
ei.image_size = [0,0];				% pixel size of image for each layer

ei.filter_sizes = [-1,-1];							% filter size for convolutional layers
ei.numFilters = [-1,-1];
ei.pool_dim = [-1,-1];


% hyperparameters for convulutional layer(s)
%ei.image_size = [ei.input_pixels,0,0];				% pixel size of image for each layer
%ei.filter_sizes = [9,0,0];							% filter size for convolutional layers
%ei.numFilters = [10,0,0];
%ei.pool_dim = [2,0,0];
% scaling parameter for l2 weight regularization penalty
ei.lambda = 0;
ei.sgd=1;

% debugging parameters
ei.random=true;
ei.debug_level=0;
ei.check=true;
ei.stop_backprop=true;

%%% setup minfuncSGD options
options = [];
options.epochs = 3;
options.minibatch = 256;
options.alpha = 0.003;
options.momentum = .95;
options.annealing=1.1;
algo_basic=1;
algo_adagrad=2;
options.algo = algo_adagrad;

ratio_of_dataset=1;
[data_train, labels_train, data_test, labels_test] = load_preprocess_mnist(ratio_of_dataset);
