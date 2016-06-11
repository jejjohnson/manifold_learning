function [X,lambda] = schroedingerEigenmap(L,V,beta,numEigs)
% schroedingerEigenmap - compute Schroedinger eigenmap
% usage: [X,lambda] = schroedingerEigenmap(L,V,beta,numEigs);
%
% arguments:
%   L (sparse NxN) - graph Laplacian
%   V (sparse NxN) - matrix of potentials 
%   beta (scalar) - multiplicative factor to generate S = L + beta*V
%   numEigs (scalar) - number of eigenvectors to return
%
%   XS (N x (numEigs+1)) - eigenvectors of Schroedinger operator
%   lambda ((numEigs+1) x (numEigs+1)) - corresponding eigenvalues
%
%   Note: numEigs+1 eigenvectors returned because if only nondiagonal
%   potentials are used, the vector of all 1's should be the first
%   eigenvector, and this eigenvector is usually discarded from subsequent
%   analysis.
%

% author: Nathan D. Cahill
% email: nathan.cahill@rit.edu
% date: 14 October 2013

D = spdiags(spdiags(L,0),0,size(L,1),size(L,1));

% compute more eigenvectors than necessary due to convergence of eigs
[X,lambda] = eigs(L+beta*V,D,round(1.5*(numEigs+1)),'SA');

% don't return first eigenvector/eigenvalue
X = X(:,2:numEigs+1);
lambda = lambda(2:numEigs+1,2:numEigs+1);