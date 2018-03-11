function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, X_val, y_val)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector lambda_vec contains the different lambda parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               you the errors obtained after training with 
%               lambda = lambda_vec(i)
%
% Note: You can loop over lambda_vec with the following:
%
%       for i = 1:length(lambda_vec)
%           lambda = lambda_vec(i);
%           % Compute train / val errors when training linear 
%           % regression with regularization parameter lambda
%           % You should store the result in error_train(i)
%           % and error_val(i)
%           ....
%           
%       end
%
%

% Size of training set
m = size(X, 1);

% Size of cross-validation set
m_val = size(X_val, 1);

for i = 1:length(lambda_vec)
    % Set regularization parameter
    lambda_i = lambda_vec(i);
    
    % Calculate linear regression coefficients using chosen regularization
    % parameter
    theta_i = trainLinearReg(X, y, lambda_i);
    
    % Compute training error
    error_train(i) = 1/(2*m)*norm(X*theta_i - y)^2;
    
    % Compute cross-validation error
    error_val(i) = 1/(2*m_val)*norm(X_val*theta_i - y_val)^2;
end








% =========================================================================

end
