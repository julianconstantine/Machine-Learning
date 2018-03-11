function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% Compute hypotheses function h(x; theta) = 1/(1 + exp(-theta'*x))
h = sigmoid(X*theta);

% Compute cost function
% J(theta) = (1/m)*sum[from i = 1 to m: y(i)*log(h(x(i); theta)) + (1 -
% y(i))*log(1 - h(x(i); theta))]

% for i = 1:m
%     J = J + 1/m*(y(i)*log(h(i)) + (1 - y(i))*log(1 - h(i)));
% end

% Compute cost function using vector operations 
J = -1/m*sum(y.*log(h) + (1-y).*log(1-h));

% Compute gradient (i.e. Jacobian derivative) of the cost function
% dJ/d[theta(j) for j = 1 to N] = (1/m)*sum[from i = 1 to m: (x(x(i); theta) 
% - y(i))*x(i)]

% for j = 1:length(theta)
%     for i = 1:m
%         grad(j) = grad(j) + 1/m*(h(i) - y(i))*X(i, j);
%     end
% end

% Compute gradient using vector operations
grad = 1/m*X'*(h - y);


% =============================================================

end
