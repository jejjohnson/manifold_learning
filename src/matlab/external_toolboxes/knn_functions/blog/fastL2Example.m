%%=========================================================
%%  Fast Euclidean Distance Calculation
%   
%   This script demonstrates the use of matrix multiplication to quickly
%   calculate the Euclidean distance between a large number of vectors.
%
% $Author: ChrisMcCormick $    $Date: 2014/08/22 22:00:00 $    $Revision: 1.0 $

%%=========================================================
%%  Generate Dataset
%%=========================================================
 
clear;
 
n = 256;        % 'n' is the vector length.
m = 10000;      % 'm' is the number of input vectors.
k = 100;        % 'k' is the number of models to compare against.

% Generate random input vectors.
X = rand(m, n);

% Generate random models to compare against.
C = rand(k, n);


%%=========================================================
%%  Sum-of-squared-differences approach
%%=========================================================

% Measure the time.
tic();

% Create a matrix to hold the distances between each data point and
% each model.
dists = zeros(m, k);

% For each model...
for (i = 1 : k)

    % Subtract model i from all data points.
    diffs = bsxfun(@minus, X, C(i, :));

    % Take the square root of the sum of squared differences.
    dists(:, i) = sqrt(sum(diffs.^2, 2));
end

elapsed = toc();
fprintf('Sum-of-squared-differences took %.3f seconds.\n', elapsed);
if (exist('OCTAVE_VERSION')) fflush(stdout); end

dists1 = dists;


%%=========================================================
%%  Matrix-multiply approach
%%=========================================================

tic();

% Calculate the sum of squares for all input vectors and
% for all cluster centers / models.
%
% Matrix dimensions:
%   X  [m  x  n]
%  XX  [m  x  1]
%   C  [k  x  n]
%  CC  [1  x  k]
XX = sum(X.^2, 2);
CC = sum(C.^2, 2)';

% Calculate the dot product between each input vector and
% each cluster center / model. 
%  
% Matrix dimensions:
%   X  [m  x  n]    
%   C  [k  x  n]
%   C' [n  x  k]
%  XC  [m  x  k]
XC = X * C';

% Calculate the Euclidean distance between all input vectors in X
% and all clusters / models in C using the following equation:
%
%   z = sqrt(||x||^2 - 2xc' + ||c||^2)
%
%  Step 1: Subtract the column vector XX from every column of XC.
%  Step 2: Add the row vector CC to every row of XC.
%
% Matrix dimensions:
%     XX  [m  x  1]
%     XC  [m  x  k]
%     CC  [1  x  k] 
%  dists  [m  x  k]
%  
dists = sqrt(bsxfun(@plus, CC, bsxfun(@minus, XX, 2*XC)));

elapsed = toc();
fprintf('Dot-product took %.3f seconds.\n', elapsed);
if (exist('OCTAVE_VERSION')) fflush(stdout); end

dists2 = dists;

% Make sure the resulting distances are nearly identical.
fprintf('Difference in result: %f\n', sum(abs(dists1(:) - dists2(:))));
if (exist('OCTAVE_VERSION')) fflush(stdout); end