%{ 
This is a sample script for the Laplacian Eigenmaps algorithm.
I am going to walk through and deconstruct the script piece-by-piece.

Indian Pines Data Available from:
% http://www.ehu.es/ccwintco/uploads/6/67/Indian_pines_corrected.mat
% http://www.ehu.es/ccwintco/uploads/6/67/Indian_pines_gt.mat

%}
clear all; close all; clc;

%% Load Sample Data

load('saved_data/img_data')

%#########################################
%% Construct Adjacency Matrix - Test
%#########################################

%% K NEAREST NEIGHBORS - standard spectral
clear options
options.type = 'standard';
options.saved = 0;
options.k = 4;

W = Adjacency(imgVec, options);


figure(1);
spy(W, 1e-15)

%% K NEAREST NEIGHBORS - standard spatial

% create the spatial neighbors matrix
[x, y] = meshgrid(1:size(img,2), 1:size(img,1));

pData = [x(:) y(:)];

clear options
options.type = 'standard';
options.saved = 0;
options.k = 4;

W = Adjacency(pData, options);

figure(2);
spy(W, 1e-15)

%% K NEAREST NEIGHBORS - similarity
data = [ones(10,1); zeros(20,1); ones(10,1); zeros(20,1)];

clear options
options.type = 'similarity';
options.saved = 0;


W = Adjacency(data, options);

figure(2);
spy(W, 1e-15)

%% K NEAREST NEIGHBORS - dissimilarity
data = [ones(10,1); zeros(20,1); 2*ones(10,1); zeros(20,1)];

clear options
options.type = 'dissimilarity';
options.saved = 0;


W = Adjacency(data, options);

figure(2);
spy(W, 1e-15)

% K NEAREST NEIGHBORS - label-propagation


