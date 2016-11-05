% Mu-Alpha Parameter Estimation


% Clear Variables
clear all; close all; clc;

% dataset
dataset = 'vcu';
rng('default');
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



%=================================\
%% Manifold Alignment Parameters
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

% spatial spectral potential matrix options
PotentialOptions = [];
PotentialOptions.type = 'spaspec';
PotentialOptions.clusterSigma = 1;
PotentialOptions.clusterkernel = 'heat';
PotentialOptions.weightSigma = 1;

% spatial adjacency matrix options
SpatialAdjacency.type = 'standard';
SpatialAdjacency.nn_graph = 'knn';
SpatialAdjacency.k = 4;
SpatialAdjacency.kernel = 'heat';
SpatialAdjacency.sigma = 1;
SpatialAdjacency.saved = 0;

% save spatial adjacency matrix options
PotentialOptions.SpatialAdjacency = SpatialAdjacency;

% save potential options
Options.PotentialOptions = PotentialOptions;

% Manifold Alignment Options
AlignmentOptions = [];

AlignmentOptions.nComponents = 'default';
AlignmentOptions.printing = 0;
AlignmentOptions.lambda = 0;

%===============
% Experiment Options
%==================================%


embedding = [];
rng('default');         % reproducibility

Options.labelPrct = {.1, .1};
Options.trainPrct = {0.4, 0.4};
Data = getdataformat(ImageData, Options);


% manifold alignment options
AlignmentOptions.mu = .4;
AlignmentOptions.type = 'ssma';    
AlignmentOptions.printing = 1;

% save Alignment options
Options.AlignmentOptions = AlignmentOptions;

% Manifold Alignment
projectionFunctions = manifoldalignment(Data, Options);

% Embedding
embedding.ssma = manifoldalignmentprojections(Data, projectionFunctions, 'ssma');

% statistics
ClassOptions = [];
ClassOptions.dimStep = 4;

ClassOptions.method = 'lda';
stats.ssmalda = alignmentclassification(Data, embedding.ssma, ClassOptions); 
        

% manifold alignment options
AlignmentOptions.mu = .04;
AlignmentOptions.type = 'wang';    

% save Alignment options
Options.AlignmentOptions = AlignmentOptions;

% Manifold Alignment
projectionFunctions = manifoldalignment(Data, Options);

% Embedding
embedding.wang = manifoldalignmentprojections(Data, projectionFunctions, 'wang');
            
% statistics
ClassOptions = [];
ClassOptions.dimStep = 4;

ClassOptions.method = 'lda';
stats.wanglda = alignmentclassification(Data, embedding.wang, ClassOptions);


% manifold alignment options
AlignmentOptions.mu = .4;
AlignmentOptions.type = 'sema'; 
AlignmentOptions.alpha = 20;

% save Alignment options
Options.AlignmentOptions = AlignmentOptions;

% Manifold Alignment
projectionFunctions = manifoldalignment(Data, Options);

% Embedding
embedding.sema = ...
    manifoldalignmentprojections(Data, projectionFunctions, 'ssse');

% statistics
ClassOptions = [];
ClassOptions.dimStep = 4;

ClassOptions.method = 'lda';
stats.semalda = alignmentclassification(Data, embedding.sema, ClassOptions);

%%
Options = [];
Options.nDims = {40, 50};
Options.algo = 'wang';
Options.method = 'lda';
Stats = alignmentplots(Data, embedding, Options);

        

    
    
