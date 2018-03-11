function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%

% Length of vector X (number of training examples)
m = numel(X);

% You need to return the following variables correctly.
% X_poly is an m x p matrix
X_poly = zeros(m, p);

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% 

% Store X.^j for j = 1 to p in the jth column of X_poly
for j = 1:p
    X_poly(:, j) = X.^j;
end




% =========================================================================

end
