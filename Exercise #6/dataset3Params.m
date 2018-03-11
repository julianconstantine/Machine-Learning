function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
% C = 0;
% sigma = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

errors = zeros(length(C_vec), length(sigma_vec));

for i = 1:length(C_vec)
    for j = 1:length(sigma_vec)
        C_i = C_vec(i);
        sigma_j = sigma_vec(j);
        
        model_ij = svmTrain(X, y, C_i, @(x1, x2) gaussianKernel(x1, x2, sigma_j));

        y_predict_ij = svmPredict(model_ij, Xval);
        
        errors(i, j) = mean(double(y_predict_ij ~= yval));
    end
end

% I am finding that the values of C and sigma that minimize the error
% function are C = 1, sigma = 0.1
[~, ind] = min(errors(:));
[min_i, min_j] = ind2sub([size(errors, 1) size(errors, 2)], ind);

C = C_vec(min_i); 

sigma = sigma_vec(min_j);

% =========================================================================

end
