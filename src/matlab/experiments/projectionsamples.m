% Different Number of Samples - 
%
%
% Data 
% ----
% * Indian Pines
% * Pavia
% * Houston
%
%

% Clear Environment
clear all; close all; clc;

%============================%
% Gather Data for experiment %
%============================%

dataset = 'indianpines';

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
        img_gt = importdata('PaviaU_gt.mat');

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
lpp_options.knn = options;                      % LPP Algorithm
sep_options.spectral_nn = options;              % SEP Algorithm



%=================================%
% Eigenvalue Decompsition Options %
%=================================%

options = [];
options.n_components = 150;
options.constraint = 'degree';

% save eigenvalue decomposition options
lpp_options.embedding = options;                % LPP Algorithm
sep_options.embedding = options;                % SEP Algorithm


%==========================%
% Spatial Spectral Options %
%==========================%

options = [];
options.image = img;

% save partial labels options
sep_options.ss = options;                       % SEP Algorithm

% schroedinger eigenmaps type
sep_options.type = 'spaspec';                   % SEP Algorithm

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Choose Samples Parameter
nSamples = .1:.1:.8;
traintestoptions = [];

% choose classificaiton parameters
ClassOptions = [];


count = 1;
% Perform Kernel Eigenmap method
for iSample = nSamples
    
    % split the data into training and testing
    traintestoptions.trainPrct = iSample;
    [XTrain, YTrain, XTest, YTest] = traintestsplit(...
        imgVec, gtVec, traintestoptions);
    
    % Kernel Eigenmap Method - LPP
    [embedding, ~] = LocalityPreservingProjections(...
        XTrain, lpp_options);
    
    % Project the Training and Testing samples
    XTrain = XTrain * embedding;        % Training Data
    XTest = XTest * embedding;          % Testing Data
    
    % save data for later
    save_path = 'H:\Data\saved_data\projection_samples\';
    save_str = char([save_path sprintf('%s_lpp_sampl%2.f', dataset, iSample*100)]);
    save(save_str, 'XTrain', 'XTest', 'YTrain', 'YTest', 'iSample');
    
%     % Kernel Eigenmap Method - SEP
%     [embedding, ~] = SchroedingerEigenmapProjections(...
%         XTrain, sep_options);
%     
%     % Project the Training and Testing samples
%     XTrain = XTrain * embedding;        % Training Data
%     XTest = XTest * embedding;          % Testing Data
%     
%     
%     % save data for later
%     save_path = 'H:\Data\saved_data\projection_samples\';
%     save_str = char([save_path sprintf('%s_sep_sampl%d', dataset, iSample*10)]);
%     save(save_str, 'XTrain', 'XTest', 'YTrain', 'YTest', 'iSample');
    
    count = count + 1;
    
end
    

