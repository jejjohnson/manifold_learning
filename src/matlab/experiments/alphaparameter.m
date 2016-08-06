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
addpath('H:\Data\Images\RS\IndianPines\');      % specific to PC

img = importdata('Indian_pines_corrected.mat');
imgGT = importdata('Indian_pines_gt.mat');

% remove path from matlab space
rmpath('H:\Data\Images\RS\IndianPines\');  

%=========================================%
% Reorder and Rescale data into 2-D Array %
%=========================================%

[numRows, numCols, numSpectra] = size(img);
scfact = mean(reshape(sqrt(sum(img.^2,3)), numRows*numCols,1));
img = img./scfact;
imgVec = reshape(img, [numRows*numCols numSpectra]);
gtVec = reshape(imgGT, [numRows*numCols 1]);

%==================================%
% Spectral K-NN Graph Construction %
%==================================%

options = [];
options.type = 'standard';
options.saved = 1;
options.k = 20;

% save spectral knn options 
sep_options.spectral_nn = options;              % SEP Algorithm
se_options.spectral_nn = options;               % SE Algorithm


%=================================%
% Eigenvalue Decompsition Options %
%=================================%

options = [];
options.n_components = 150;
options.constraint = 'degree';

% save eigenvalue decomposition options
lpp_options.embedding = options;                % LPP Algorithm
sep_options.embedding = options;                % SEP Algorithm
le_options.embedding = options;                 % LE Algorithm
sep_options.embedding = options;                % SE Algorithm


%==========================%
% Spatial Spectral Options %
%==========================%

options = [];
options.image = img;

% save partial labels options
sep_options.ss = options;                       % SEP Algorithm
se_options.ss = options;                        % SE Algorithm

% schroedinger eigenmaps type
sep_options.type = 'spaspec';                   % SEP Algorithm
se_options.type = 'spaspec';                    % SE Algorithm


%========================%
% Choose Alpha Parameter %
%========================%

alpha = logspace(-1,2,10);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Perform Kernel Eigenmap Projection Method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
embedding = cell(size(alpha));
count = 0;

for ialpha = alpha
    
    % use alpha parameter
    count = count + 1;


    %=================================%
    % Spatial K-NN Graph Construction %
    %=================================%

    options = [];
    options.k = 4;
    options.saved = 0;
    options.alpha = ialpha;

    % save spatial knn options
    sep_options.spatial_nn = options;               % SEP Algorithm

    %====================================%
    % Schroedinger Eigenmap Projections %
    %====================================%

    tic;
    projections = SchroedingerEigenmapProjections(imgVec, sep_options);
    embedding{count} = imgVec * projections;             % project data
    time.sep = toc;
    
    
    
end


%=========================
% Classification 
%===================%
classOptions = [];
classOptions.trainPrct = .10;
classOptions.experiment = 'statsdims';
classOptions.method = 'svm';
classOptions.gtVec = gtVec;

statssep = [];

for icount = 1:count
    
    [statssep{icount}] = classexperiments(embedding{icount}, classOptions);
end

% save the statistics for later
save_path = 'H:\Data\saved_data\alpha_results\indian_pines\sep_';
save_str = char([ save_path sprintf('alpha_k%d', 20)]);
save(save_str, 'embedding', 'statssep')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Perform Kernel Eigenmap Method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
count = 0;
embedding = cell(size(alpha));

for ialpha = alpha
    
    % use alpha parameter
    count = count + 1;


    %=================================%
    % Spatial K-NN Graph Construction %
    %=================================%

    options = [];
    options.k = 4;
    options.saved = 0;
    options.alpha = ialpha;

    % save spatial knn options
    se_options.spatial_nn = options;               % SEP Algorithm

    %====================================%
    % Schroedinger Eigenmap Projections %
    %====================================%

    tic;
    embedding{count} = SchroedingerEigenmaps(imgVec, se_options);
    time.se = toc;
    
    
    
end


%=========================
%% Classification 
%===================%
classOptions = [];
classOptions.trainPrct = .10;
classOptions.experiment = 'statsdims';
classOptions.method = 'svm';
classOptions.gtVec = gtVec;

statsse = [];

for icount = 1:count
    
    [statsse{icount}] = classexperiments(embedding{icount}, classOptions);
end

%% save the statistics for later
statsse = statssep;
save_path = 'H:\Data\saved_data\alpha_results\indian_pines\se_';
save_str = char([ save_path sprintf('alpha_k%d', 20)]);
save(save_str, 'embedding', 'statsse')

% %========================%
% %% Schroedinger Eigenmaps %
% %========================%
% 
% tic;
% embedding.se = SchroedingerEigenmaps(imgVec, se_options);
% time.se = toc;
% 
% classOptions = [];
% classOptions.trainPrct = .10;
% classOptions.experiment = 'statsdims';
% classOptions.method = 'svm';
% classOptions.gtVec = gtVec;
% [statssep] = classexperiments(embedding, classOptions);
% 
% %% save the statistics for later
% save_path = 'H:\Data\saved_data\class_results\indian_pines\sep_';
% save_str = char([ save_path sprintf('k%d', 20)]);
% save(save_str, 'embedding', 'statssep')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot Results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%









