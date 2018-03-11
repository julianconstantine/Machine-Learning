function [mu, sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%

% mean() and std() both default to column-wise calculations
mu = mean(X);

% Just using std() does not work because MATLAB's built-in standard
% deviation calculatuor uses the formula 1/(m-1)*sum((x-mu)^2) and we are
% using the formula 1/m*sum((x-mu)^2). Thus, we have to manually re-adjust
% the coefficient. 
% sigma2 = (m-1)/m*std(X).^2;

% Or, you can specify the 'w' parameter in std()
% w=0 is normalization by m-1, w=1 is normalization by m
sigma2 = std(X, 1).^2;

% =============================================================


end
