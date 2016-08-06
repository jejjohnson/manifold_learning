% Experiment Script


% Clear Variables
clear all; close all; clc;

% dataset
dataset = 'vcu';

switch lower(dataset)
    case 'vcu'
        % Import Images
        load H:\Data\Images\RS\VCU\vcu_images.mat;
        
        ImageData = [];
        ImageData{1}.img = vcu_data(1).img;             % 400m image
        ImageData{2}.img = vcu_data(2).img;             % 2000m image

        ImageData{1}.gt = vcu_data(1).img_gt2;      % 400m image ground truth
        ImageData{2}.gt = vcu_data(2).img_gt2;      % 2000m image ground truth
        
        
        % Image preprocessing
        for iImage = 1:numel(ImageData)

            % image preprocessing
            ImageData{iImage}.img = normalizeimage(ImageData{iImage}.img ); 

            % convert image to array
            [ImageData{iImage}.imgVec, ImageData{iImage}.dims] = imgtoarray(ImageData{iImage}.img);
    
            % convert ground truth to array
            [ImageData{iImage}.gtVec, ImageData{iImage}.gtdims] = imgtoarray(ImageData{iImage}.gt);
        end

        
    otherwise
        error('Unrecognized dataset chosen.');
end


% Get Data in appropriate form
Options.trainPrct = .1;
Options.labelPrct = .1;
Data = getdataformat(ImageData, Options);

%=================================\
%% Manifold Alignment
%=================================%


% Semisupervised Manifold Alignment
Options = [];

% adjacency matrix options
AdjacencyOptions = [];
AdajcencyOptions.type = 'standard';
AdjacencyOptions.nn_graph = 'knn';
AdjacencyOptions.k = 20;
AdjacencyOptions.kernel = 'heat';
AdjacencyOptions.sigma = 1;
AdjacencyOptions.saved = 0;

Options.AdjacencyOptions = AdjacencyOptions;

% potentialmatrix options
PotentialOptions = [];
PotentialOptions.type = 'spaspec';
PotentialOptions.clusterSigma = 1;
PotentialOptions.clusterkernel = 'heat';
PotentialOptions.weightSigma = 1;

SpatialAdjacency.type = 'standard';
SpatialAdjacency.nn_graph = 'knn';
SpatialAdjacency.k = 4;
SpatialAdjacency.kernel = 'heat';
SpatialAdjacency.sigma = 1;
SpatialAdjacency.saved = 0;

PotentialOptions.SpatialAdjacency = SpatialAdjacency;

Options.PotentialOptions = PotentialOptions;


% manifold alignment options
AlignmentOptions = [];
AlignmentOptions.type = 'sssema'; %'ssma';
AlignmentOptions.nComponents = 'default';
AlignmentOptions.printing = 0;
AlignmentOptions.lambda = 0;
AlignmentOptions.mu = .2;
AlignmentOptions.alpha = .2;


Options.AlignmentOptions = AlignmentOptions;


embedding = manifoldalignment(Data, Options);








    