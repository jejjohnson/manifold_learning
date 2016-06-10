%{ 
This is a sample script for the Laplacian Eigenmaps algorithm.
I am going to walk through and deconstruct the script piece-by-piece.

Indian Pines Data Available from:
% http://www.ehu.es/ccwintco/uploads/6/67/Indian_pines_corrected.mat
% http://www.ehu.es/ccwintco/uploads/6/67/Indian_pines_gt.mat

%}
clear all; close all; clc;

%% Load Indian Pines Data

% add file path of where my data is located
addpath('/media/eman/Emans HDD/Data/Images/RS/IndianPines')

% load the images
load('Indian_pines_corrected.mat');
load('Indian_pines_gt.mat');

% set them to variables
img = indian_pines_corrected;
gt = indian_pines_gt;

% clear the path as well as the datafiles 
clear indian*
rmpath('/media/eman/Emans HDD/Data/Images/RS/IndianPines')

%##########################################
%% Reorder and Rescale data into a 2D array
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
%% Spatial Positions of the data
%##########################################

% create a meshgrid function that will be the size of the spatial
% domain of the image. the meshgrid function basically captures the
% coordinates of the image
[x, y] = meshgrid(1:dims.cols, 1:dims.rows);

% now that we have the coordinates, create a matrix that has coordinates
% laid out side by side like an imgVec. think of it like a long vector 
% with 2 features: the x-direction and y-direction
pData = [x(:) y(:)];


%#########################################
%% Construct Adjacency Matrix
%#########################################

% K NEAREST NEIGHBORS
knnVal = 20;
distType = 'euclidean';     % distance measure between neighbors
try 
    
    % try loading in previous data in subdirectory
    load('saved_data/knn_data');
    disp('Found previous computation in saved files');
catch
    
    % if not, will manually do implementation and save it for later
    warning('No previous implementations...calculating knn');
    
    % first, find the 20-nearest neighbors

    % JIT Implementation
    tic;
    [knn.idxJIT, knn.distJIT] = knnJIT(imgVec, imgVec, knnVal);
    knn.timeJIT = toc;
    
    % print time elapsed
    fprintf('Knn Search - JIT: %.3f.s\n', knn.timeJIT)

    % MATLAB Implementation
    tic;
    [knn.idx, knn.dist]=knnsearch(imgVec, imgVec,'k',...
        knnVal+1,'Distance',distType);
    knn.time = toc;
    
    % print time elapsed
    fprintf('Knn Search - JIT: %.3f.s\n', knn.time)
    
    knn.idx = knn.idx(:, 2:end);
    knn.dist = knn.dist(:, 2:end);
    
    % save knn implementations for later
    save('saved_data/knn_data.mat', 'knn')
end

% DISTANCE KERNELS

% element-wise operation 
% 1) pairwise distances squared
% 2) - sol/sigma_parameter
% 3) exponentiate
w = exp(-(knn.dist.^2)./(.5.^2));

% SPARSE ADJACENCY MATRIX

%{ 
MATLAB Function - sparse(i,j,v,m,n) where
    * i is the x-coordinate
    * j is the y-coordinate
    * v is the entry
    * m, n is the (m x n) size of the matrix

i : repeat the length of nodes for the graph from 1 to k
j : insert the knn indices found
v : insert the knn weighted distances found
m , n : number of nodes in the graph (n samples from data)
%}

i = repmat((1:dims.nodes)',[1 knnVal]);
j = knn.idx;
v = w;
[m, n] = deal(dims.nodes);

W = sparse(i,j,v, m, n);

W = max(W,W');      % make the matrix symmetric

% take a peek at what the graph looks like
% figure(1);
% spy(W, 1e-16)
% title('Nearest Neighbor Graph, W');

%#########################################
%% Construct Graph Laplacian
%#########################################

% Diagonal Degree Matrix

%{ 
spdiags(B,d,m,n) where
    * B is a vector of elements
    * d is the place within the diagonal range (0 is the exact diagonal)
    * m, n is the (m x n) size of the matrix
B : sum the W matrix along the columns
d : 0 (we want it in the center)
m , n : number of samples of imgVec
%}
D = spdiags(sum(W, 2), 0, dims.nodes, dims.nodes);

% SPECTRAL LAPLACIAN MATRIX (D-W)
L = D - W;



%############################################
%% Eigenvalue Decomposition
%############################################

% number of components we want to keep
n_components = 150;

tic;

% perform the eigenvalue decomposition (we want the smallest ones)
[embedding, lambda] = eigs(L,D,round(1.5*(n_components+1)),'sm');
time = toc;

% disregard the smallest eigenvalue because it should be zero
embedding = embedding(:,2:n_components+1);
lambda = diag(lambda); lambda = lambda(2:n_components+1);

fprintf('Eigenvalue Decomposition: %.3f.\n', time)



%#############################################################
%% Experiment III - Tuia et al. LDA w/ Assessment
%#############################################################


test_dims = (1:10:150);
% testing and training

lda_class = [];
svm_class = [];

h = waitbar(0, 'Initializing waitbar...');

for dim = test_dims
    
    waitbar(dim/n_components)
    % # of dimensions
    XS = embedding(:,1:dim);
    
    [Xtr, Ytr, Xts, Yts , ~, ~] = ppc(XS, gt_Vec, .25);
    
    % Classifiaction - LDA
    [Ypred, err] = classify(Xts, Xtr, Ytr);
    
    % Assessment
    Results = assessment(Yts, Ypred, 'class');
    
    lda_class = [lda_class; Results.OA];
    
%     % Classification - SVM
%     Ypred = svmClassify(Xtr, Ytr, Xts);
%     
%      % Assessment - SVM
%     Results = assessment(Yts, Ypred, 'class');
%     
%     svm_class = [svm_class; (Results.Kappa)];
    
end

close(h)

figure('Units', 'pixels', ...
    'Position', [100 100 500 375]);
hold on;

% plot lines
hLECV = line(test_dims, lda_class);
hLE = line(test_dims, svm_class);

% set some first round of line parameters
set(hLECV, ...
    'Color',        'r', ...
    'LineWidth',    2);
set(hLE, ...
    'Color',        'b', ...
    'LineWidth',    2);

hTitle = title('LE + LDA - Indian Pines');
hXLabel = xlabel('d-Dimensions');
hYLabel = ylabel('Correct Rate');

hLegend = legend( ...
    [hLECV, hLE], ...
    'Kappa Coefficient',...
    'Overall Accuracy',...
    'location', 'NorthWest');

% pretty font and axis properties
set(gca, 'FontName', 'Helvetica');
set([hTitle, hXLabel, hYLabel],...
    'FontName', 'AvantGarde');
set([hXLabel, hYLabel],...
    'FontSize',10);
set(hTitle,...
    'FontSize'  ,   12,...
    'FontWeight',   'bold');

set(gca,...
    'Box',      'off',...
    'TickDir',  'out',...
    'TickLength',   [.02, .02],...
    'XMinorTick',   'on',...
    'YMinorTick',   'on',...
    'YGrid',        'on',...
    'XColor',       [.3,.3,.3],...
    'YColor',       [.3, .3, .3],...
    'YTick'     ,   0:0.1:1,...
    'XTick'     ,   0:10:100,...
    'LineWidth' ,   1);



break
%#############################################################
%% Experiment I - Dimension versus Class Accuracy
%#############################################################

% eigenvalue decomposition parameters
test_dims = (1:5:50);

% part 1 - do LE + LDA w/ Cross-Validation
error_rates.le_cv = zeros(size(test_dims));

% part 2 - do LE + LDA 
error_rates.le = zeros(size(test_dims));

% part 3 - do LDA w/ Cross-Validation
error_rates.lda_cv = zeros(size(test_dims));

% part 4 - do LDA
error_rates.lda = zeros(size(test_dims));



%--------------------------------------------
% Training vs. Testing - No CrossValidation
%-------------------------------------------- 

trainRatio = .1;        % Training Idx percentage
testRatio = .9;         % Testing Idx percentage
valRatio = 0.0;         % Validation Idx percentage

% get the training and testing indices
[trainidx, ~, testidx] = divideint(size(gt_Vec,1), .25,.0,.75);

% prefer a (samples x 1) vector 
trainidx = trainidx'; testidx = testidx';

%-------------------------------------------- ---
% Training vs. Testing - 10-Fold CrossValidation
%-------------------------------------------- ---

% number of folds
k_folds = 10;

% cross validation indices
crossvalidx = crossvalind('kfold', gt_Vec, k_folds);

% classification performance object (testing w/ ground truth vec)
cp.le_cv = classperf(gt_Vec);       % LE w/ CV performance measurer
cp.lda_cv = classperf(gt_Vec);      % LE performance measurer

cp.lda = classperf(gt_Vec);         % LDA w/ cv performance measurer
cp.le = classperf(gt_Vec);          % LDA w/o cv performance measurer

%-------------------------------------------- ---
%% Experiment 1a,b - Cross-Validation
%-------------------------------------------- ---

% keep track of dimensions
dim_count = 1;

for dim = test_dims
    
    % # of dimensions
    XS = embedding(:,1:dim);
    
    % Classifiaction - Cross-Validation
    
    % initialize the performance measurer
    cp.le_cv = classperf(gt_Vec);       % LE w/ CV performance measurer
    
    for i = 1:10
        test = (crossvalidx == i); train = ~ test;
        class = classify(XS(test,:), XS(train,:), ...
            gt_Vec(train,:));
        classperf(cp.le_cv, class, test);
    end
    
    % store error rates
    error_rates.le_cv(dim_count) = cp.le_cv.CorrectRate;
    
    % Classification - No Cross-Validation
    
    % initialize the performance measurer
    cp.le = classperf(gt_Vec);       % LE w/ CV performance measurer
    
    class = classify(XS(testidx,:), XS(trainidx,:),...
        gt_Vec(trainidx,:));
    classperf(cp.le, class, testidx);
    
    % store error rates
    error_rates.le(dim_count) = cp.le.CorrectRate;
    
    
    % count next iteration
    dim_count = dim_count + 1;
end



%################################################
%% Experiment I - Plot Results
%################################################
    

figure('Units', 'pixels', ...
    'Position', [100 100 500 375]);
hold on;

% plot lines
hLECV = line(test_dims, error_rates.le_cv);
hLE = line(test_dims, error_rates.le);

% set some first round of line parameters
set(hLECV, ...
    'Color',        'r', ...
    'LineWidth',    2);
set(hLE, ...
    'Color',        'b', ...
    'LineWidth',    2);

hTitle = title('LE + LDA - Indian Pines');
hXLabel = xlabel('d-Dimensions');
hYLabel = ylabel('Correct Rate');

hLegend = legend( ...
    [hLECV, hLE], ...
    'Cross-Validation',...
    'No Cross-Validation',...
    'location', 'NorthWest');

% pretty font and axis properties
set(gca, 'FontName', 'Helvetica');
set([hTitle, hXLabel, hYLabel],...
    'FontName', 'AvantGarde');
set([hXLabel, hYLabel],...
    'FontSize',10);
set(hTitle,...
    'FontSize'  ,   12,...
    'FontWeight',   'bold');

set(gca,...
    'Box',      'off',...
    'TickDir',  'out',...
    'TickLength',   [.02, .02],...
    'XMinorTick',   'on',...
    'YMinorTick',   'on',...
    'YGrid',        'on',...
    'XColor',       [.3,.3,.3],...
    'YColor',       [.3, .3, .3],...
    'YTick'     ,   0:0.1:1,...
    'XTick'     ,   0:10:100,...
    'LineWidth' ,   1);



%#############################################################
%% Experiment II - Cahill SVM
%#############################################################



%%

C = cahill_svm(embedding, gt, dims.rows, dims.cols);
