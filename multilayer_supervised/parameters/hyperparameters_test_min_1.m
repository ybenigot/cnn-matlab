fprintf('hyperparameters TEST MIN 1\n');

%% populate ei with the network architecture to train
% ei is a structure you can use to store hyperparameters of the network
% the architecture specified below should produce  100% training accuracy
% You should be able to try different network architectures by changing ei
% only (no changes to the objective function code)


% dimension of input features
ei.input_pixels = 28; 
ei.input_dim = ei.input_pixels * ei.input_pixels; %784
% number of output classes
ei.output_dim = 20*20*2;
ei.input_depth=1; % number of input planes 1 for B&W, 3 for RGB
% which type of activation function to use in hidden layers
% feel free to implement support for only the logistic sigmoid function
% this hyper parameter shall have same size as layer_sizes 
% using 'sigmoid' implies that two functions are defined : sigmoid, and its derivative sigmoid_d
ei.activation_functions = ['relu'];
ei.cost_function = 'squared_cost'; % possibilities are square_cost and cross_entropy_cost 

ei.layer_type= [convolve];
% sizes of all hidden layers and the output layer ; number of features for convolutional layer
ei.layer_sizes = [ei.output_dim];

% hyperparameters for convulutional layer(s)
ei.image_size = [ei.input_pixels,0];				% pixel size of image for each layer
ei.filter_sizes = [9];							% filter size for convolutional layers
ei.numFilters = [2];
ei.pool_dim = [-1];

% scaling parameter for l2 weight regularization penalty
ei.lambda = 0.0;
ei.sgd=1;

% debugging parameters
ei.random=true;
ei.debug_level=0;
ei.check=true;
ei.stop_backprop=false;

%%% setup minfuncSGD options
options = [];
options.epochs = 3;
options.minibatch = 128;
options.alpha = 0.001;
options.annealing=1.3;
options.momentum = 0.9;
algo_basic=1;
algo_adagrad=2;
options.algo = algo_adagrad;

fprintf('TESTING VALUES\n')
%data_train(:,1)=reshape(1:28*28,28*28,1)/784;
%IM = data_train(:,1);
ratio_of_dataset=1;
[data_train, labels_train, data_test, labels_test] = load_preprocess_mnist(ratio_of_dataset);


