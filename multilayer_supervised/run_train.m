% runs training procedure for supervised multilayer network
% softmax output layer with cross entropy loss function

diary record/diary;
diary on
more off

%% setup environment
% experiment information
% a struct containing network layer sizes etc
ei = [];
connected=1;
convolve=2;
pooling=3;

% add common directory to your path for
% minfunc and mnist data helpers
addpath ../common;
%addpath(genpath('../common/minFunc_2012/minFunc'));
addpath parameters;
addpath preprocess;
addpath funcs/image;
addpath funcs/activation;
addpath funcs/cost;
addpath funcs/utils;
addpath funcs/porting2matlab;

%% load mnist data
global train_stats;
train_stats=[];

hyperparameters_cnn_sgd_4; % ----------------------------------------------

ei % display hyperparameters
ei.activation_functions
ei.pool_types


%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack,ei);

fprintf('parameters size : %d,%d\n',size(params));

displayData(data_train(:,1:min(100,size(data_train,2)))',ei.input_pixels,'displaying first images...\n',4);

%% run training
fprintf('run training...\n');
start1=cputime;
%fflush(stdout);

if ei.sgd==0
	fprintf('full gradient descent\n');
	% use Stanford common/minFunc2012 for this
	%[opt_params,opt_value,exitflag,output] = minFunc(@supervised_nn_cost,...
	%    params,options,ei, data_train, labels_train,false);
elseif ei.sgd==1
	fprintf('stochastic gradient descent\n');
	[opt_params] = minFuncSGD(@supervised_nn_cost,params,ei,data_train,labels_train,options);
else
	fprintf('invalid sgd\n');
end	

fprintf('end training after %d s\n',cputime-start1);

save(strcat('record/train-',date));

fprintf('dataset size :    %d,%d\n',size(data_train));
fprintf('parameters size : %d,%d\n',size(params));

fprintf('compute accuracy...\n');

%% compute accuracy on the test and train set
m10=min(size(data_train,2),5000);
[~, ~, pred] = supervised_nn_cost( opt_params, ei, data_train(:,1:m10), [], true, false, true); % also show images
[~,pred] = max(pred);
acc_train = mean(pred'==labels_train(1:m10));
fprintf('for %d samples, train accuracy: %f\n', m10, acc_train);

m20=min(size(data_test,2),5000);
[~, ~, pred] = supervised_nn_cost( opt_params, ei, data_test(:,1:m20), [], true);
[~,pred] = max(pred);
acc_test = mean(pred'==labels_test(1:m20));
fprintf('for %d samples, test accuracy: %f\n', m20, acc_test);

%printf('training statistics\n');
%train_stats

% check the result visually
showimages;	

close all;

% show trained parameters
displayParameters(opt_params, ei, 1);

close all;

% show training cost function graph
m1=1:size(train_stats,1);
%plot(m1,train_stats(:,1),"r",m1,train_stats(:,2),"g",m1,train_stats(:,3),"b",m1,train_stats(:,4),"y");
plot(m1,train_stats(:,1),'b');




diary off
more on
