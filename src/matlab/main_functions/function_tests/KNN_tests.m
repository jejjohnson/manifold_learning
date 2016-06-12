% Priminary stuff
clear all; close all; clc;
% add file path of where my data is located
addpath('H:\Data\Images\RS\IndianPines')
% addpath('/media/eman/Emans HDD/Data/Images/RS/IndianPines')

% load the images
load('Indian_pines_corrected.mat');
load('Indian_pines_gt.mat');

% set them to variables
img = indian_pines_corrected;
gt = indian_pines_gt;

% clear the path as well as the datafiles 
clear indian*
addpath('H:\Data\Images\RS\IndianPines')
% rmpath('/media/eman/Emans HDD/Data/Images/RS/IndianPines')

%##########################################
%{
    We do this in order to make the data more workable as well as rescale
    the data to the unit norm. We can't do processing as is with the 
    3D cube. So we make a 2D cube where it's basically a long vector
    of values with features representative as each dimension.
%}

% get the dimensions of the image
[dims.rows, dims.cols, dims.spectra] = size(img);

% find the squared values along the 3rd dimension of the array
scfact = sum(img.^2,3);

% reshape the summation into an image vector (rows*cols x dims)
scfact = reshape(scfact, dims.rows*dims.cols, 1);

% take the mean of that 
scfact = mean(scfact);

% divide the entire image by that mean
img = img./scfact;

fprintf('The max of the image is: %.2d.\n', max(img(:)))
fprintf('The min of the image is: %.2d.\n', min(img(:)))

% create the image vector
imgVec = reshape(img, [dims.rows*dims.cols dims.spectra]);
gt_Vec = reshape(gt, [dims.rows*dims.cols 1]);
dims.nodes = size(imgVec,1);

%##########################################

%#########################################

% K NEAREST NEIGHBORS
knnVal = 20;
distType = 'euclidean';     % distance measure between neighbors


%% Tested Methods
try
    % try loading in previous data in subdirectory
    load('saved_data/knn_data');
    disp('Found previous computation in saved files...');
    % print time elapsed
    fprintf('KD Tree Search - MATLAB: %.3f.s\n', knn.timekd)
    fprintf('Knn Search - MATLAB: %.3f.s\n', knn.time)
    fprintf('Knn Search - MATLAB: %.3f.s\n', knn.timepll)
catch
    
    % if not, will manually do implementation and save it for later
    warning('No previous implementations...calculating knn...could take a while...');
    
    % first, find the 20-nearest neighbors
    
    %-------------------------------------
    % MATLAB Implementation - KDTree Method
    %---------------------------------------
    tic;
    
    % initialize the KD Tree Model
    KDModel = KDTreeSearcher(imgVec, 'Distance', distType);
    
    % query the vector for k nearest neighbors
    [knn.idxkd, knn.distkd]=knnsearch(KDModel, imgVec,'k',...
        knnVal+1);
    knn.timekd = toc;
    
    % print time elapsed
    fprintf('KD Tree Search - MATLAB: %.3f.s\n', knn.timekd)
    
    knn.idxkd = knn.idxkd(:, 2:end);
    knn.distkd = knn.distkd(:, 2:end);    
    %-------------------------------------
    % MATLAB Implementation - Brute force
    %---------------------------------------
    tic;
    [knn.idx, knn.dist]=knnsearch(imgVec, imgVec,'k',...
        knnVal+1,'Distance',distType);
    knn.time = toc;
    
    % print time elapsed
    fprintf('Knn Search - MATLAB: %.3f.s\n', knn.time)
    
    knn.idx = knn.idx(:, 2:end);
    knn.dist = knn.dist(:, 2:end);
    
    % save knn implementations for later
    disp('...saving knn data for later usage...');
    save('saved_data/se_knn_data.mat', 'knn')
    
    %---------------------------------------
    % MATLAB Implementation - Parallel
    %---------------------------------------
    tic;

    % initialize the KD Tree Model
    num_cores = 8;
    [knn.distpll, idx.distpll] = knn_parallel(imgVec, num_cores);
    time = toc;

    % print time elapsed
    fprintf('Parallel KNN - MATLAB: %.3f.s\n', time)
    % save knn implementations for later
    disp('...saving knn data for later usage...');
    save('saved_data/knn_data.mat', 'knn')

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Subfunctions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

