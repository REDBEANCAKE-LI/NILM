function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% Initialize variables
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% extend feature matrix
X = [ones(m,1) X];

% compute activation probability of hidden layer nodes
hid_prob = sigmoid(X*Theta1');

% extend hidden layer matrix 
hid_prob = [ones(m,1) hid_prob];

% compute activation probability of output layer nodes
out_prob = sigmoid(hid_prob*Theta2');

y_extend = zeros(m,num_labels);
for i=1:m,
    y_extend(i,y(i)) = 1;    %process the labels
end;

% compute cost function
J = -1/m*sum(sum(y_extend.*log(out_prob)+(1-y_extend).*log(1-out_prob)))+lambda/(2*m)*(sum(nn_params.^2)-sum(Theta1(:,1).^2)-sum(Theta2(:,1).^2));

% implement back propogation
delta3 = out_prob - y_extend;
delta2 = (delta3*Theta2(:,2:end)).*sigmoidGradient(X*Theta1');
D1 = 0; D2 = 0;
for i=1:m,
    D1 = D1 + delta2(i,:)'*X(i,:);
    D2 = D2 + delta3(i,:)'*hid_prob(i,:);
end;
Theta1_grad = 1/m*D1;
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m*Theta1(:,2:end);
Theta2_grad = 1/m*D2;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m*Theta2(:,2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
