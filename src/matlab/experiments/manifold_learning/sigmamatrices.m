% Different Sigma Matrices - 
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

% Gather data for experiment
indianPinesPath = 'H:\Data\Images\RS\IndianPines\';
paviaPath = 'H:\Data\Images\RS\Pavia\';

% add specific directory to matlab space
addpath('H:\Data\Images\RS\IndianPines\');      % specific to PC

img = importdata('Indian_pines_corrected.mat');
img_gt = importdata('Indian_pines_gt.mat');

% remove path from matlab space
rmpath('H:\Data\Images\RS\IndianPines\');  

%=========================================%
% Reorder and Rescale data into 2-D Array %
%=========================================%

[numRows, numCols, numSpectra] = size(img);
scfact = mean(reshape(sqrt(sum(img.^2,3)), numRows*numCols,1));
img = img./scfact;
imgVec = reshape(img, [numRows*numCols numSpectra]);


%=========================================%
% k-Nearest Neighbors - Indian Pines      %
%=========================================%
sigmas = logspace(-1,2,10);

options.type = 'standard';
options.k = 20;                     % 20 k-nearest neighbors
options.saved = 1;                  % save the knn parameters calculated



count = 1;

for iSigma = sigmas
    tic;
    options.sigma = iSigma;
    
    % find the knns w/ function
    
    [varString, ~] = Adjacency(imgVec, options);
    saveString = ['H:\Data\saved_data\sigma_results\IndianPines' ...
        char(sprintf('_sigma%d', options.k, count))];
    
    save(saveString, 'varString', 'sigmas', 'options');  % save variable to file
    options.saved = 1;              % use saved knn parameters calculated
    count = count + 1;              % loop counter
    toc;
end

% save the results to folder



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PAVIA UNIVERSITY
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Clear Environment
clear all; close all; clc;

% add specific directory to matlab space
addpath('H:\Data\Images\RS\Pavia\');      % specific to PC

img = importdata('PaviaU.mat');
img_gt = importdata('PaviaU_gt.mat');

% remove path from matlab space
rmpath('H:\Data\Images\RS\Pavia\');  

%=========================================%
% Reorder and Rescale data into 2-D Array %
%=========================================%

[numRows, numCols, numSpectra] = size(img);
scfact = mean(reshape(sqrt(sum(img.^2,3)), numRows*numCols,1));
img = img./scfact;
imgVec = reshape(img, [numRows*numCols numSpectra]);


%=========================================%
% k-Nearest Neighbors - Indian Pines      %
%=========================================%
sigmas = logspace(-1,2,10);

options.type = 'standard';
options.k = 20;                     % 20 k-nearest neighbors
options.saved = 1;                  % save the knn parameters calculated



count = 1;

for iSigma = sigmas
    tic;
    options.sigma = iSigma;
    
    [varString, ~] = Adjacency(imgVec, options);
    saveString = ['H:\Data\saved_data\sigma_results\Pavia' ...
        char(sprintf('_sigma%d', options.k, count))];
    
    save(saveString, 'varString', 'sigmas', 'options');  % save variable to file
    options.saved = 1;              % use saved knn parameters calculated
    count = count + 1;              % loop counter
    toc;
end



    