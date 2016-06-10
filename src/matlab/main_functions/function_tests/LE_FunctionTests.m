%{ 
This is a sample script for the Laplacian Eigenmaps algorithm.
I am going to walk through and deconstruct the script piece-by-piece.

Indian Pines Data Available from:
% http://www.ehu.es/ccwintco/uploads/6/67/Indian_pines_corrected.mat
% http://www.ehu.es/ccwintco/uploads/6/67/Indian_pines_gt.mat

%}
clear all; close all; clc;

%% Load Indian Pines Data Adjacency Matrix


load('../../saved_data/le_adjacency.mat');
%--------------------------------------------------------------------------
% Test the Eigenvalue Decomposition - No Inputs
%--------------------------------------------------------------------------


[embedding, lambda] = LaplacianEigenmaps(W);

%--------------------------------------------------------------------------
% Test the Eigenvalue Decomposition - No Inputs; Identity Matrix
%--------------------------------------------------------------------------
options.constraint = 'identity';
[embedding, lambda] = LaplacianEigenmaps(W, options);
%--------------------------------------------------------------------------
% Test the Eigenvalue Decomposition - No Inputs; Degree Matrix
%--------------------------------------------------------------------------
options.constraint = 'degree';
[embedding, lambda] = LaplacianEigenmaps(W, options);

%--------------------------------------------------------------------------
% Test the Eigenvalue Decomposition - No Inputs; K-Scaling Matrix
%--------------------------------------------------------------------------

options.constraint = 'k_scaling';
[embedding, lambda] = LaplacianEigenmaps(W, options);
%--------------------------------------------------------------------------
% Test the Eigenvalue Decomposition - No Inputs; Dissimilarity Matrix
%--------------------------------------------------------------------------
options.constraint = 'dissimilarlity';
[embedding, lambda] = LaplacianEigenmaps(W, options);

%--------------------------------------------------------------------------
% Test the Eigenvalue Decomposition - Using RSVD Decomposition
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
% Test the Eigenvalue Decomposition - Using RSVD Decomposition
%--------------------------------------------------------------------------
