% Different Adjacency Matrices - 
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
kNNs = [1, :5:100];

options.type = 'standard';
options.saved = 0;



count = 1;
tic;
for ik = kNNs
    options.k = ik;
    % find the knns w/ function
    
    [varString, ~] = Adjacency(imgVec, options);
    saveString = char(sprintf('experiments/saved_data/adjacency/IndianPines_k%d', ik));
    
    save(saveString, 'varString');
    
    count = count + 1;
    
end

% save the results to folder

toc;

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
clear W
% kNNs = [1, 5:5:100];
kNNs = [35:5:100];
options.type = 'standard';
options.saved = 0;



count = 1;
tic;
for ik = kNNs
    options.k = ik;
    
    [varString, ~] = Adjacency(imgVec, options);
    saveString = char(sprintf('experiments/saved_data/adjacency/Pavia_k%d', ik));
    
    save(saveString, 'varString');
    
    count = count + 1;
    
end

% save the results to folder
save experiments/saved_data/adjacencyPavia W
toc;
    