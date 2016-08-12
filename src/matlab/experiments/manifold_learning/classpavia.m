% Classification Experiment - I
% 
%
% Data
% ----
% * Indian Pines HyperSpectral Image
% * Pavia Hyperspectral Image
%
% Algorithms
% ----------
% * Laplacian Eigenmaps
% * Locality Preserving Projections
% * Schroedinger Eigenmaps
% * Schroedinger Eigenmap Projections
%
% Output
% ------
% * OA      - Overall Accuracy
% * AA      - Average Accuracy
% * k       - Kappa Coefficient
% * AP      - Average Precision
% * Recall  - Recall
% * Dim     - Dimension
% * F1      - F1 Statistic
% * Time    - Time For Complete Algorithm (secs)
%
% Figures 
% -------
% * Classification Maps (all Algorithms)
% * k-NN Parameter Estimations
% * Potential Trade off Parameter Alpha
% * Potential Trade off Parameter Beta
% * % Samples vs. Dimensions
% * k-NN vs. Dimensions vs. % Samples
% * 
%
% Tables
% ------
% * Best Parameters Chosen
% * Statistical Significance
%
% Information
% -----------
% Author: Juan Emmanuel Johnson
% Email: jej2744@rit.edu
% Date: 27th July, 2016
%

% Clear Environment
clear all; close all; clc;

%============================%
% Gather Data for experiment %
%============================%

% add specific directory to matlab space
addpath('H:\Data\Images\RS\Pavia\');      % specific to PC

img = importdata('PaviaU.mat');
imgGT = double(importdata('PaviaU_gt.mat'));

% remove path from matlab space
rmpath('H:\Data\Images\RS\pavia\');  

%=========================================%
% Reorder and Rescale data into 2-D Array %
%=========================================%

[numRows, numCols, numSpectra] = size(img);
scfact = mean(reshape(sqrt(sum(img.^2,3)), numRows*numCols,1));
img = img./scfact;
imgVec = reshape(img, [numRows*numCols numSpectra]);
gtVec = reshape(imgGT, [numRows*numCols 1]);




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Laplacian Eigenmaps
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%==================================%
% Spectral K-NN Graph Construction %
%==================================%

options = [];
options.type = 'standard';
options.saved = 0;
options.k = 20;
options.sigma = 1;

le_options.knn = options;                       % save options

%=================================%
% Eigenvalue Decompsition Options %
%=================================%

options = [];
options.n_components = 50;
options.constraint = 'degree';

le_options.embedding = options;                 % save options

%=================================%
% Laplacian Eigenmaps Algorithm %
%=================================%

tic;
le.embedding = LaplacianEigenmaps(imgVec, le_options);


%=================================%
%% Classification Results
%=================================%

classOptions = [];
classOptions.nDims = 50;
classOptions.trainPrct = .10;
classOptions.experiment = 'bestresults';
classOptions.method = 'svm';
classOptions.gtVec = gtVec;
[Stats.le] = classexperiments(le.embedding, classOptions);


% Get time
Stats.le.time = toc;


%=================================%
%% Classification Plots
%=================================%

% Image predictions
classOptions = [];
classOptions.nDims = 50;
classOptions.trainPrct = .10;
classOptions.experiment = 'imagepredictions';
classOptions.method = 'svm';
classOptions.gtVec = gtVec;
imgPred.le = classexperiments(le.embedding, classOptions);

% Classification Plots
imgPred.le = reshape(imgPred.le, [numRows, numCols, 1]);
Options = [];
Options.type = 'learning';
Options.hsi = 'pavia';
Options.algo = 'le';
plotclassmaps(imgPred.le, img, imgGT, Options);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Schroedinger Eigenmaps
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%==================================%
% Spectral K-NN Graph Construction %
%==================================%

options = [];
options.type = 'standard';
options.saved = 0;
options.k = 20;
options.sigma = 1;

se_options.knn = options;                       % save options

%=================================%
% Spatial K-NN Graph Construction %
%=================================%

options = [];
options.k = 4;
options.saved = 0;

% save spatial knn options
se_options.spatial_nn = options;                % SE Algorithm


%=================================%
% Eigenvalue Decompsition Options %
%=================================%

options = [];
options.n_components = 50;
options.constraint = 'degree';

% save eigenvalue decomposition options
se_options.embedding = options;                % LPP Algorithm


%==========================%
% Spatial Spectral Options %
%==========================%

options = [];
options.image = img;

% save partial labels options
se_options.ss = options;                        % SE Algorithm

% schroedinger eigenmaps type
se_options.type = 'spaspec';                    % SE Algorithm

%========================%
% Schroedinger Eigenmaps %
%========================%


tic;
se.embedding = SchroedingerEigenmaps(imgVec, se_options);


%=================================%
% Classification Results
%=================================%

classOptions = [];
classOptions.nDims = 50;
classOptions.trainPrct = .10;
classOptions.experiment = 'bestresults';
classOptions.method = 'svm';
classOptions.gtVec = gtVec;
[Stats.se] = classexperiments(se.embedding, classOptions);


% Get time
Stats.se.time = toc;


%=================================%
% Classification Plots
%=================================%

% Image predictions
classOptions = [];
classOptions.nDims = 50;
classOptions.trainPrct = .10;
classOptions.experiment = 'imagepredictions';
classOptions.method = 'svm';
classOptions.gtVec = gtVec;
imgPred.se = classexperiments(se.embedding, classOptions);

% Classification Plots
imgPred.se = reshape(imgPred.se, [numRows, numCols, 1]);
Options = [];
Options.type = 'learning';
Options.hsi = 'pavia';
Options.algo = 'se';
plotclassmaps(imgPred.se, img, imgGT, Options);
%% Locality Preserving Projections
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%==================================%
% Spectral K-NN Graph Construction %
%==================================%

options = [];
options.type = 'standard';
options.saved = 0;
options.k = 20;
options.sigma = 1;

lpp_options.knn = options;                       % save options

%=================================%
% Eigenvalue Decompsition Options %
%=================================%

options = [];
options.n_components = 50;
options.constraint = 'degree';

lpp_options.embedding = options;                 % save options

%=================================%
% Laplacian Eigenmaps Algorithm %
%=================================%

tic;
lpp.embedding = LocalityPreservingProjections(imgVec, lpp_options);
lpp.embedding = imgVec * lpp.embedding;


%=================================%
% Classification Results
%=================================%

classOptions = [];
classOptions.nDims = 50;
classOptions.trainPrct = .10;
classOptions.experiment = 'bestresults';
classOptions.method = 'svm';
classOptions.gtVec = gtVec;
[Stats.lpp] = classexperiments(lpp.embedding, classOptions);


% Get time
Stats.lpp.time = toc;

%=================================%
% Classification Plots
%=================================%

% Image predictions
classOptions = [];
classOptions.nDims = 50;
classOptions.trainPrct = .10;
classOptions.experiment = 'imagepredictions';
classOptions.method = 'svm';
classOptions.gtVec = gtVec;
imgPred.lpp = classexperiments(lpp.embedding, classOptions);

% Classification Plots
imgPred.lpp = reshape(imgPred.lpp, [numRows, numCols, 1]);
Options = [];
Options.type = 'learning';
Options.hsi = 'pavia';
Options.algo = 'lpp';
plotclassmaps(imgPred.lpp, img, imgGT, Options);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Schroedinger Eigenmap Projections 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%==================================%
% Spectral K-NN Graph Construction %
%==================================%

options = [];
options.type = 'standard';
options.saved = 0;
options.k = 20;
options.sigma = 1;

sep_options.knn = options;                       % save options

%=================================%
% Spatial K-NN Graph Construction %
%=================================%

options = [];
options.k = 4;
options.saved = 0;

% save spatial knn options
sep_options.spatial_nn = options;                % sep Algorithm


%=================================%
% Eigenvalue Decompsition Options %
%=================================%

options = [];
options.n_components = 50;
options.constraint = 'degree';

% save eigenvalue decomposition options
sep_options.embedding = options;                % LPP Algorithm


%==========================%
% Spatial Spectral Options %
%==========================%

options = [];
options.image = img;

% save partial labels options
sep_options.ss = options;                        % sep Algorithm

% schroedinger eigenmaps type
sep_options.type = 'spaspec';                    % sep Algorithm

%========================%
% Schroedinger Eigenmaps %
%========================%


tic;
sep.embedding = SchroedingerEigenmapProjections(imgVec, sep_options);
sep.embedding = imgVec * sep.embedding;

%=================================%
% Classification Results
%=================================%

classOptions = [];
classOptions.nDims = 50;
classOptions.trainPrct = .10;
classOptions.experiment = 'bestresults';
classOptions.method = 'svm';
classOptions.gtVec = gtVec;
[Stats.sep] = classexperiments(sep.embedding, classOptions);


% Get time
Stats.sep.time = toc;


%=================================%
% Classification Plots
%=================================%

% Image predictions
classOptions = [];
classOptions.nDims = 50;
classOptions.trainPrct = .10;
classOptions.experiment = 'imagepredictions';
classOptions.method = 'svm';
classOptions.gtVec = gtVec;
imgPred.sep = classexperiments(sep.embedding, classOptions);

% Classification Plots
imgPred.sep = reshape(imgPred.sep, [numRows, numCols, 1]);
Options = [];
Options.type = 'learning';
Options.hsi = 'pavia';
Options.algo = 'sep';
plotclassmaps(imgPred.sep, img, imgGT, Options);




%% Best Results

classNames = {...
    'Trees       ';...
    'Asphalt     '; ...
    'Bitumen     '; ...
    'Gravel      '; ...
    'Metal Sheets'; ...
    'Shadow      '; ...
    'Bricks      '; ...
    'Meadows     '; ...
    'Bare Soil   '};
Stats.classNames = classNames;

Options = [];
Options.exp = 'manifold';
bestclassresults(Stats, Options);











