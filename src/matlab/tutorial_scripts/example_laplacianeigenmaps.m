% clear workspace
clear; close all force; clc;

% Get Example Data
ImageData = getexampleimg;

% Construct default Adjacency Matrix
GeometricModel = Adjacency(ImageData.imageVector);

%% Find the nearest neighbors
Options.k = 10;
[idx, dist] = GeometricModel.nearestneighbors(ImageData.imageVector, Options);

%% Find the distance via kernels
w = GeometricModel.distancekernel(dist);

%% Construct the Adajency Matrix
W = GeometricModel.constructadjacency(ImageData.imageVector, idx, w);

%% Complete Geometric Model Class
W = GeometricModel.getadjacency;