%% initialize
clear; close all; clc;

%% constants
input_layer_size = 736;	% the number of pixels of each example
hidden_layer_size = 30;	% number of hidden units
num_labels = 5;	% number of labels, from 1 to 5

%% ******* load and separate data *************
% load data
fprintf('loading data...\n');
X = load('features.mat').values;
y = load('labels.mat').values;

% separate training set and test set
size_train = round(size(X, 1) * 0.7);
size_test = size(X, 1) - size_train;
sel = randperm(size(X, 1));
sel_train = sel(1:size_train);
sel_test = sel((size_train + 1):end);
% test sel_train + sel_test
%fprintf('size_train + size_test = %f\n', size_train + size_test);
X_train = X(sel_train, :);
X_test = X(sel_test, :);
y_train = y(sel_train, :);
y_test = y(sel_test, :);

fprintf('pause\n');
pause;

%% ******* initialize weights *********************
fprintf('initializing neural network parameters...\n');
initial_theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% unroll parameters
initial_nn_params = [initial_theta1(:) ; initial_theta2(:)];

fprintf('pause\n');
pause;

%% ******* train NN ****************************
fprintf('training neural network...\n');

% set options
options = optimset('MaxIter', 50);

% set lambda
lambda = 0.1;

% create 'short hand' for the cost functio to be minimized
costFunction = @(p) nnCostFunction(p, ...
				   input_layer_size, ...
				   hidden_layer_size, ...
				   num_labels, ...
				   X_train, y_train, lambda);

% train
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% obtain theta1 and theta2 back from nn_params
theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
		 hidden_layer_size, (input_layer_size + 1));
theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))): end), ...
		 num_labels, (hidden_layer_size + 1));

fprintf('pause\n');
pause;

%% ************ predict ******************************
pred1 = predict(theta1, theta2, X_train);
pred2 = predict(theta1, theta2, X_test);
fprintf('training set accuracy: %f\n', mean(double(pred1 == y_train)) * 100);
fprintf('test set accuracy: %f\n', mean(double(pred2 == y_test)) * 100);

fprintf('pause\n');
pause;

%% ************ show ********************************
% show 30 test cases
randlist = randperm(size(X, 1));
testlist = randlist(1:30);

fprintf('show 30 test cases...\n');
fprintf('no.\t\tlabel\t\tpredict\t\t\n');
for i=1:30
	fprintf('%d\t\t%d\t\t%d\t\t\n', testlist(i), y(testlist(i)), predict(theta1, theta2, X(testlist(i), :)));
end
