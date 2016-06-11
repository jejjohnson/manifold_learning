%{ 
This is a sample script for the Laplacian Eigenmaps algorithm.
I am going to walk through and deconstruct the script piece-by-piece.

Indian Pines Data Available from:
% http://www.ehu.es/ccwintco/uploads/6/67/Indian_pines_corrected.mat
% http://www.ehu.es/ccwintco/uploads/6/67/Indian_pines_gt.mat

%}
clear all; close all; clc;

%% Load Indian Pines Data

% add file path of where my data is located
addpath('/media/eman/Emans HDD/Data/Images/RS/IndianPines')

% load the images
load('Indian_pines_corrected.mat');
load('Indian_pines_gt.mat');

% set them to variables
img = indian_pines_corrected;
gt = indian_pines_gt;

% clear the path as well as the datafiles 
clear indian*
rmpath('/media/eman/Emans HDD/Data/Images/RS/IndianPines')

%##########################################
%% Reorder and Rescale data into a 2D array
%##########################################
%{
    We do this in order to make the data more workable as well as rescale
    the data to the unit norm. We can't do processing as is with the 
    3D cube. So we make a 2D cube where it's basically a long vector
    of values with features representative as each dimension.
%}

% get the dimensions of the image
[dims.rows, dims.cols, dims.spectra] = size(img);

% find the squared values along the 3rd dimension of the array
scfact = sum(img.^2,3);

% reshape the summation into an image vector (rows*cols x dims)
scfact = reshape(scfact, dims.rows*dims.cols, 1);

% take the mean of that 
scfact = mean(scfact);

% divide the entire image by that mean
img = img./scfact;

fprintf('The max of the image is: %.2d.\n', max(img(:)))
fprintf('The min of the image is: %.2d.\n', min(img(:)))

% create the image vector
imgVec = reshape(img, [dims.rows*dims.cols dims.spectra]);
gt_Vec = reshape(gt, [dims.rows*dims.cols 1]);
dims.nodes = size(imgVec,1);

%##########################################
%% Spatial Positions of the data
%##########################################

% create a meshgrid function that will be the size of the spatial
% domain of the image. the meshgrid function basically captures the
% coordinates of the image
[x, y] = meshgrid(1:dims.cols, 1:dims.rows);

% now that we have the coordinates, create a matrix that has coordinates
% laid out side by side like an imgVec. think of it like a long vector 
% with 2 features: the x-direction and y-direction
pData = [x(:) y(:)];

break
%#########################################
%% Construct Adjacency Matrix
%#########################################

% K NEAREST NEIGHBORS
knnVal = 20;
distType = 'euclidean';     % distance measure between neighbors


try 
    
    % try loading in previous data in subdirectory
    load('saved_data/knn_data');
    disp('Found previous computation in saved files');
catch
    
    % if not, will manually do implementation and save it for later
    warning('No previous implementations...calculating knn');
    
    % first, find the 20-nearest neighbors

    % JIT Implementation
    tic;
    [knn.idxJIT, knn.distJIT] = knnJIT(imgVec, imgVec, knnVal);
    knn.timeJIT = toc;
    
    % print time elapsed
    fprintf('Knn Search - JIT: %.3f.s\n', knn.timeJIT)

    % MATLAB Implementation
    tic;
    [knn.idx, knn.dist]=knnsearch(imgVec, imgVec,'k',...
        knnVal+1,'Distance',distType);
    knn.time = toc;
    
    % print time elapsed
    fprintf('Knn Search - JIT: %.3f.s\n', knn.time)
    
    knn.idx = knn.idx(:, 2:end);
    knn.dist = knn.dist(:, 2:end);
    
    % save knn implementations for later
    save('saved_data/knn_data.mat', 'knn')
end

% DISTANCE KERNELS

% element-wise operation 
% 1) pairwise distances squared
% 2) - sol/sigma_parameter
% 3) exponentiate
w = exp(-(knn.dist.^2)./(.5.^2));

% SPARSE ADJACENCY MATRIX

%{ 
MATLAB Function - sparse(i,j,v,m,n) where
    * i is the x-coordinate
    * j is the y-coordinate
    * v is the entry
    * m, n is the (m x n) size of the matrix

i : repeat the length of nodes for the graph from 1 to k
j : insert the knn indices found
v : insert the knn weighted distances found
m , n : number of nodes in the graph (n samples from data)
%}

i = repmat((1:dims.nodes)',[1 knnVal]);
j = knn.idx;
v = w;
[m, n] = deal(dims.nodes);

W = sparse(i,j,v, m, n);

W = max(W,W');      % make the matrix symmetric

% take a peek at what the graph looks like
% figure(1);
% spy(W, 1e-16)
% title('Nearest Neighbor Graph, W');

%#########################################
%% Construct Graph Laplacian
%#########################################

% Diagonal Degree Matrix

%{ 
spdiags(B,d,m,n) where
    * B is a vector of elements
    * d is the place within the diagonal range (0 is the exact diagonal)
    * m, n is the (m x n) size of the matrix
B : sum the W matrix along the columns
d : 0 (we want it in the center)
m , n : number of samples of imgVec
%}
D = spdiags(sum(W, 2), 0, dims.nodes, dims.nodes);

% SPECTRAL LAPLACIAN MATRIX (D-W)
L = D - W;

%############################################
%% LaplacianEigenmaps function test
%############################################
n_components = 150;
options.n_components = n_components;

tic;
[embedding, lambda] = LaplacianEigenmaps(W, options);
time = toc;

% number of components we want to keep


tic;

fprintf('Eigenvalue Decomposition: %.3f.\n', time)
save('../saved_data/le_eigvals.mat', 'embedding', 'lambda')
%############################################
%% Eigenvalue Decomposition
%############################################
try 
    
    % try loading in previous data in subdirectory
    load('saved_data/le_eigvals');
    disp('Found previous computation in saved files...');
catch
    n_components = 150;
    options.n_components = n_components;
    
    tic;
    [embedding, lambda] = LaplacianEigenmaps(W, options);
    time = toc;
    
    % number of components we want to keep


    tic;

    fprintf('Eigenvalue Decomposition: %.3f.\n', time)
    save('saved_data/le_eigvals.mat', 'embedding', 'lambda')
end

