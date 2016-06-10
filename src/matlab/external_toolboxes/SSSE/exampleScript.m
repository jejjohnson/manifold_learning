% exampleScript.m: Provides example code for performing Spatial-Spectral 
% Schroedinger Eigenmaps for dimensionality reduction, as described in the
% papers:
%
% 1) N. D. Cahill, W. Czaja, and D. W. Messinger, "Schroedinger Eigenmaps 
% with Nondiagonal Potentials for Spatial-Spectral Clustering of 
% Hyperspectral Imagery," Proc. SPIE Defense & Security: Algorithms and 
% Technologies for Multispectral, Hyperspectral, and Ultraspectral Imagery 
% XX, May 2014. 
%
% 2) N. D. Cahill, W. Czaja, and D. W. Messinger, "Spatial-Spectral
% Schroedinger Eigenmaps for Dimensionality Reduction and Classification of
% Hyperspectral Imagery," submitted.
%
% This example script also performs classification using Support Vector
% Machines, as described in paper 2.
%
clear all; close all; clc;
%% Load Indian Pines data, available from:
% http://www.ehu.es/ccwintco/uploads/6/67/Indian_pines_corrected.mat
% http://www.ehu.es/ccwintco/uploads/6/67/Indian_pines_gt.mat
%

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

%% reorder and rescale data into 2-D array
[numRows,numCols,numSpectra] = size(img);
scfact = mean(reshape(sqrt(sum(img.^2,3)),numRows*numCols,1));
img = img./scfact;
imgVec = reshape(img,[numRows*numCols numSpectra]);

%% get spatial positions of data
[x,y] = meshgrid(1:numCols,1:numRows);
pData = [x(:) y(:)];

%% construct adjacency matrix via one of many methods

% select SSSE method you'd like to use
% options:
%   'SSSE_(SM)^(f,p)' (this is SSSE1 from paper 1)
%   'SSSE_(SM)^(p,f)'
%   'SSSE_(GB)^(f,p)' (this is SSSE2 from paper 1)
%   'SSSE_(GB)^(p,f)'

SSSEMethod = 'SSSE_(SM)^(f,p)';

% parameters
sigma = 1;
k = 20;
eta = 1;

switch SSSEMethod
    
    case 'SSSE_(SM)^(f,p)'
        [AF,idxF] = adjacencyMatrix(imgVec,[],k,sigma);
        [AP,idxP] = adjacencyMatrix([],pData,4,[],eta);
        A = AF; EData = imgVec; CData = pData; idx = idxP;
    case 'SSSE_(SM)^(p,f)'
        [AF,idxF] = adjacencyMatrix(imgVec,[],k,sigma);
        [AP,idxP] = adjacencyMatrix([],pData,4,[],eta);
        A = AP; EData = pData; CData = imgVec; idx = idxF;
    case 'SSSE_(GB)^(f,p)'
        [AF,idxF] = adjacencyMatrix(imgVec,[],k,sigma,[],[],true);
        [AP,idxP] = adjacencyMatrix([],pData,4,[],eta,true);
        A = AF; EData = imgVec; CData = pData; idx = idxP;
    case 'SSSE_(GB)^(p,f)'
        [AF,idxF] = adjacencyMatrix(imgVec,[],k,sigma,[],[],true);
        [AP,idxP] = adjacencyMatrix([],pData,4,[],eta,true);
        A = AP; EData = pData; CData = imgVec; idx = idxF;
    otherwise
        error('Invalid method.');
end

%% construct graph laplacian
numNodes = size(A,1);
D = spdiags(full(sum(A)).',0,numNodes,numNodes);
L = D - A;

%% create potential matrix to incorporate spatial connectivity
% Newer method (from paper 2):
V = schroedingerPotential(EData,CData,true,[1 1],idx);

% Older method (just from paper 1):
% V = schroedingerSpatialPotential(img,false);
% Note: second argument false = Gilles-Bowles weight (SSSE2). 
%       second argument true = Shi-Malik weight (SSSE1).

% determine scale factor that makes the potential matrix have the same
% trace as L
scVEqL = trace(L)./trace(V);

% choose value of alpha to trade off L and V
alpha = 17.78;

%% compute generalized eigenvectors
numEigs = 50;
[XS,lambdaS] = schroedingerEigenmap(L,V,alpha*scVEqL,numEigs);

% Note: for standard Laplacian Eigenmaps algorithm (Schroedinger Eigenmaps
% with no potentials), one can execute:
% [XS,lambdaS] = schroedingerEigenmap(L,spalloc(numNodes,numNodes,0),0,numEigs);
%%

gt_Vec = reshape(gt, [numRows*numCols 1]);

lda_class = [];
[Xtr, Ytr, Xts, Yts , ~, ~] = ppc(XS, gt_Vec, .25);

% Classifiaction - LDA
[Ypred, err] = classify(Xts, Xtr, Ytr);

% Assessment
Results = assessment(Yts, Ypred, 'class');

lda_class = [lda_class; Results.OA];
fprintf('LDA Classification accuracy: %.2f\n',lda_class);

C = cahill_svm(XS, gt, numRows, numCols);


%%
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
%% create training and testing data sets for classification
trainPrct = 0.10;
rng('default'); % so each script generates the same training/testing data
[trainMask,testMask,gtMask] = createTrainTestData(gt,trainPrct);
trainInd = find(trainMask);

%% predict labels using SVM classifier
labels = svmClassify(XS(trainInd,2:end),gt(trainInd),XS(:,2:end));

%% display ground truth and predicted label image
labelImg = reshape(labels,numRows,numCols);
figure; imshow(gt,[0 max(gt(:))]); title('Ground Truth Class Labels');
figure; imshow(gtMask); title('Ground Truth Mask');
figure; imshow(labelImg,[0 max(gt(:))]); title('Predicted Class Labels');
figure; imshow(labelImg.*gtMask,[0 max(gt(:))]); title('Predicted Class Labels in Ground Truth Pixels');

%% construct confusion matrix
C = confusionmat(gt(testMask&gtMask),labels(testMask&gtMask));
