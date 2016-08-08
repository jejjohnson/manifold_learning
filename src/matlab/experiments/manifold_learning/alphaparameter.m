% Parameter Exploration - Alpha
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Gather Data for experiment %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dataset = 'pavia';

switch lower(dataset)
    case 'indianpines'

        % add specific directory to matlab space
        addpath('H:\Data\Images\RS\IndianPines\');      % specific to PC

        img = importdata('Indian_pines_corrected.mat');
        imgGT = importdata('Indian_pines_gt.mat');

        % remove path from matlab space
        rmpath('H:\Data\Images\RS\IndianPines\');  
    case 'pavia'
        
        % add specific directory to matlab space
        addpath('H:\Data\Images\RS\Pavia\');      % specific to PC

        img = importdata('PaviaU.mat');
        imgGT = importdata('PaviaU_gt.mat');

        % remove path from matlab space
        rmpath('H:\Data\Images\RS\Pavia\');  
        
    otherwise
        error('Unrecognized dataset.');
end
        

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
options.k = 20;

switch lower(dataset)
    case 'indianpines'
        options.saved = 0;
    case 'pavia'
        options.saved = 0;
    otherwise
        error('Unrecognized dataset type.');
end

% save spectral knn options 
sep_options.spectral_nn = options;              % SEP Algorithm
se_options.spectral_nn = options;               % SE Algorithm


%=================================%
% Eigenvalue Decompsition Options %
%=================================%

options = [];
options.n_components = 100;
options.constraint = 'degree';

% save eigenvalue decomposition options
lpp_options.embedding = options;                % LPP Algorithm
sep_options.embedding = options;                % SEP Algorithm
se_options.embedding = options;


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


    % save spatial knn options
    sep_options.spatial_nn = options;               % SEP Algorithm
    
    
    %=================================%
    % Spatial Spectral Options %
    %=================================%
    options = [];
    options.image = img;
    options.alpha = ialpha;

    % save partial labels options
    sep_options.ss = options;                       % SEP Algorithm
    se_options.ss = options;                        % SE Algorithm

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
classOptions.method = 'lda';
classOptions.gtVec = gtVec;

statssep = [];

for icount = 1:count
    
    [statssep{icount}] = classexperiments(embedding{icount}, classOptions);
end

% save the statistics for later
switch lower(dataset)
    case 'indianpines'
        
        save_path = 'H:\Data\saved_data\alpha_results\indian_pines\sep_';
        
    case 'pavia'
        
        save_path = 'H:\Data\saved_data\alpha_results\pavia\sep_';
        
    otherwise
        
        error('Unrecognized dataset.');
end

save_str = char([ save_path sprintf('alpha%s_k%d', method, 20)]);
save(save_str, 'embedding', 'statssep', 'classOptions')


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


    % save spatial knn options
    sep_options.spatial_nn = options;               % SEP Algorithm
    
    
    %=================================%
    % Spatial Spectral Options %
    %=================================%
    options = [];
    options.image = img;
    options.alpha = ialpha;

    % save partial labels options
    se_options.ss = options;                        % SE Algorithm

    %====================================%
    % Schroedinger Eigenmap Projections %
    %====================================%

    tic;
    embedding{count} = SchroedingerEigenmaps(imgVec, se_options);
    time.se = toc;
    
    
    
end


%=========================
% Classification 
%===================%
classOptions = [];
classOptions.trainPrct = .10;
classOptions.experiment = 'statsdims';
classOptions.method = 'lda';
classOptions.gtVec = gtVec;

statsse = [];

for icount = 1:count
    
    [statsse{icount}] = classexperiments(embedding{icount}, classOptions);
end

% save the statistics for later
switch lower(dataset)
    case 'indianpines'
        
        save_path = 'H:\Data\saved_data\alpha_results\indian_pines\se_';
        
    case 'pavia'
        
        save_path = 'H:\Data\saved_data\alpha_results\pavia\se_';
        
    otherwise
        
        error('Unrecognized dataset.');
end

save_str = char([ save_path sprintf('alpha_%s_k%d', method, 20)]);
save(save_str, 'embedding', 'statsse', 'classOptions')








