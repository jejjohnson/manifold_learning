% exampleSuperPixelScript.m: Provides example code for performing SLIC
% superpixel-based dimensionality reduction followed by SVM-based 
% classification, as described in the paper:
%
% X. Zhang, S. E. Chew, Z. Xu, and N. D. Cahill, "SLIC Superpixels for 
% Efficient Graph-Based Dimensionality Reduction of Hyperspectral Imagery,"
% Proc. SPIE Defense & Security: Algorithms and Technologies for 
% Multispectral, Hyperspectral, and Ultraspectral Imagery XXI, April 2015. 
%
% This code requires two packages that must be added to the MATLAB path:
% 
% 1) The MATLAB version of the VLFeat library, available at:
% http://www.vlfeat.org/
%
% 2) The Spectral-Spatial Schroedinger Eigenmaps library, available at the 
% MATLAB File Exchange, File No. 45908:
% http://www.mathworks.com/matlabcentral/fileexchange/45908-spatial-spectral-schroedinger-eigenmaps
%

%% Load Indian Pines data, available from:
% http://www.ehu.es/ccwintco/uploads/6/67/Indian_pines_corrected.mat
% http://www.ehu.es/ccwintco/uploads/c/c4/Indian_pines_gt.mat
%

load('Indian_pines_corrected.mat');
load('Indian_pines_gt.mat');

img = indian_pines_corrected;
gt = indian_pines_gt;

clear indian*

%% reorder and rescale data into 2-D array
[numRows,numCols,numSpectra] = size(img);
scfact = mean(reshape(sqrt(sum(img.^2,3)),numRows*numCols,1));
img = img./scfact;
imgVec = reshape(img,[numRows*numCols numSpectra]);

%% compute superpixels
disp('Computing SLIC Superpixels...');
tic; spSegs = vl_slic(single(img),8,0.01); toc
numSuperpixels = double(max(spSegs(:)))+1;

%% display image of superpixels
[sx,sy]=vl_grad(double(spSegs), 'type', 'forward') ;
s = find(sx | sy) ;

imgColor = img(:,:,[29 15 12]);
imgColor = uint8(255*(imgColor - min(imgColor(:)))./(max(imgColor(:))-min(imgColor(:))));
imgS = imgColor; imgS([s s+numel(imgColor(:,:,1)) s+2*numel(imgColor(:,:,1))]) = 0;
figure; imshow(imgS);
title('Indian Pines with SLIC Superpixels');

%% compute features at each superpixel
meanSpectra = zeros(numSuperpixels,numSpectra);
meanPosition = zeros(numSuperpixels,2);

% arrays for row, col pixel position
[r,c] = ndgrid(1:numRows,1:numCols);

% loop over each superpixel
for ii = 1:numSuperpixels
    
    % create mask for+ the i-1st superpixel
    mask = (spSegs==(ii-1));
    
    % loop through image layers to compute average spectral values
    for jj = 1:numSpectra
        imgLayer = img(:,:,jj);
        pixelSpectra = imgLayer(mask);
        meanSpectra(ii,jj) = mean(pixelSpectra);
    end
    
    % compute average row and column position
    meanPosition(ii,1) = mean(r(mask));
    meanPosition(ii,2) = mean(c(mask));
    
end

%% perform dimensionality reduction using superpixels as input

% specify dimensionality reduction method. Can be one of: 'SM', 'GB',
% 'HZYZ', 'BE', 'SSSE'
dimRedMethod = 'SSSE';

% specify number of eigenvectors
numEigs = 50;

switch dimRedMethod
    case 'SM' % Shi-Malik
        
        % construct adjacency matrix
        rad = max(numRows,numCols)/10;
        sigmaF = 0.1;
        sigmaP = 100;
        fprintf('Constructing Shi-Malik Adjacency Matrix...\n');
        tic; A = adjacencyMatrixShiMalikSuperpixels(meanSpectra,[],meanPosition,rad,sigmaF,sigmaP); toc
        
        % construct graph laplacian
        numNodes = size(A,1);
        D = spdiags(full(sum(A)).',0,numNodes,numNodes);
        L = D - A;
        
        % find first few dimensions of Laplacian eigenmap subspace
        fprintf('Computing Laplacian Eigenmap for Shi-Malik...\n');
        tic; [YS,lambdaS] = schroedingerEigenmap(L,spalloc(numNodes,numNodes,0),0,numEigs); toc;
        
    case 'GB' % Gillis Bowles
        
        % construct adjacency matrix
        rad = max(numRows,numCols)/10;
        sigmaF = 0.1;
        sigmaP = 100;
        fprintf('Constructing Gillis-Bowles Adjacency Matrix...\n');
        tic; A = adjacencyMatrixGillisBowlesSuperpixels(meanSpectra,[],meanPosition,rad,sigmaF,sigmaP); toc
        
        % construct graph laplacian
        numNodes = size(A,1);
        D = spdiags(full(sum(A)).',0,numNodes,numNodes);
        L = D - A;
        
        % find first few dimensions of Laplacian eigenmap subspace
        fprintf('Computing Laplacian Eigenmap for Gillis-Bowles...\n');
        tic; [YS,lambdaS] = schroedingerEigenmap(L,spalloc(numNodes,numNodes,0),0,numEigs); toc;

    case 'HZYZ'
        
        % construct adjacency matrix
        k = 20;
        sigmaF = 0.1;
        sigmaP = 100;
        fprintf('Constructing HZYZ Adjacency Matrix...\n');
        tic; A = adjacencyMatrixHZYZSuperpixels(meanSpectra,meanPosition,k,sigmaF,sigmaP); toc

        % construct graph laplacian
        numNodes = size(A,1);
        D = spdiags(full(sum(A)).',0,numNodes,numNodes);
        L = D - A;
        
        % find first few dimensions of Laplacian eigenmap subspace
        fprintf('Computing Laplacian Eigenmap for HZYZ...\n');
        tic; [YS,lambdaS] = schroedingerEigenmap(L,spalloc(numNodes,numNodes,0),0,numEigs); toc;

    case 'BE'
        
        % construct adjacency matrices
        fprintf('Constructing Benedetto-E Adjacency Matrix using spectral data (beta = 1)...\n');
        sigmaF = 1;
        k = 20;
        sigmaP = max(numRows,numCols)/10;
        A1 = adjacencyMatrix(meanSpectra,[],k,sigmaF,[],[],false);
        fprintf('Constructing Benedetto-E Adjacency Matrix using spatial data (beta = 0)...\n');
        A0 = adjacencyMatrix([],meanPosition,4,[],sigmaP,true);
        
        % construct graph laplacians
        numNodes = size(A0,1);
        D0 = spdiags(full(sum(A0)).',0,numNodes,numNodes);
        D1 = spdiags(full(sum(A1)).',0,numNodes,numNodes);
        L0 = D0 - A0;
        L1 = D1 - A1;
        
        % find first few dimensions of Laplacian eigenmap subspaces
        fprintf('Computing Laplacian Eigenmap for beta = 0...\n');
        tic; [YS0,lambdaS0] = schroedingerEigenmap(L0,spalloc(numNodes,numNodes,0),0,numEigs+1); toc;
        
        fprintf('Computing Laplacian Eigenmap for beta = 1...\n');
        tic; [YS1,lambdaS1] = schroedingerEigenmap(L1,spalloc(numNodes,numNodes,0),0,numEigs+1); toc;
        
        % fuse eigenvectors
        alpha = 42;
        YS = [YS0(:,2:(1+alpha)),YS1(:,2:(numEigs-alpha+1))];
                
    case 'SSSE'
        
        % construct adjacency matrix using spectral data
        fprintf('Constructing Adjacency Matrix using spectral data...\n');
        sigma = 1;
        k = 20;
        eta = max(numRows,numCols)/10;
        A = adjacencyMatrix(meanSpectra,[],k,sigma,[],[],false);
        [~,idx] = adjacencyMatrix([],meanPosition,4,[],eta,true);

        % construct graph laplacian
        numNodes = size(A,1);
        D = spdiags(full(sum(A)).',0,numNodes,numNodes);
        L = D - A;
        
        % create potential matrix to incorporate spatial connectivity
        V = schroedingerPotential(meanSpectra,meanPosition,true,[sigma eta],idx);
        
        % determine scale factor that makes the potential matrix have the same
        % trace as L
        scVEqL = trace(L)./trace(V);
        
        % choose value of alpha to trade off L and V
        alpha = 0.1;
        
        % find first few dimensions of Schroedinger eigenmap subspace
        fprintf('Computing Schroedinger Eigenmap for SSSE...\n');
        tic; [YS,lambdaS] = schroedingerEigenmap(L,V,alpha*scVEqL,numEigs);toc
                
    otherwise
        error('Dimensionality reduction method not recognized.');
end

%% interpolate eigenvector components for each pixel
fprintf('Interpolating eigenvectors to pixel resolution...\n');
tic;XInterp = interpSpectraSupPix(img,spSegs,meanPosition,1,YS);toc
XS = reshape(XInterp,[numRows*numCols,numEigs]);

%% create training and testing data sets for classification
trainPrct = 0.010;
rng('default'); % so each script generates the same training/testing data
[trainMask,testMask,gtMask] = createTrainTestData(gt,trainPrct);
trainInd = find(trainMask);

%% predict labels using SVM classifier
labels = svmClassify(XS(trainInd,2:end),gt(trainInd),XS(:,2:end));

%% display ground truth and predicted label image
gtClasses = uint8(255*ind2rgb(gt,hsv(16))).*repmat(uint8(gtMask),[1 1 3]) + ...
    255*repmat(uint8(~gtMask),[1 1 3]);
gtClassesPad = padarray(gtClasses,[1 1 0]);
figure; imshow(gtClassesPad); title('Ground Truth Class Labels');

labelImg = reshape(labels,numRows,numCols);
imgClasses = uint8(255*ind2rgb(labelImg,hsv(16)));
imgClassesPad = padarray(imgClasses,[1 1],0);
alphaData = padarray(double(gtMask)*(3/4) + 1/4,[1 1],1);
figure; h = imshow(imgClassesPad); set(h,'alphaData',alphaData);
title('Predicted Class Labels');

%% construct confusion matrix
C = confusionmat(gt(testMask&gtMask),labels(testMask&gtMask));
[BC,R,kappa,kappaVar] = binaryClassificationResults(C);

% compute various classification performance measures
OA = trace(C)/sum(C(:));
AA = nanmean(R(:,11));
AP = nanmean(R(:,12));
ASe = nanmean(R(:,13));
ASp = nanmean(R(:,14));

%% display results
fprintf('\nPer-Class Accuracy\n');
for i = 1:16
    fprintf('Class %02.0f:\t\t\t\t\t%6.4f\n',i,R(i,11));
end
fprintf('\nPer-Class Precision\n');
for i = 1:16
    fprintf('Class %02.0f:\t\t\t\t\t%6.4f\n',i,R(i,12));
end
fprintf('\nPer-Class Sensitivity\n');
for i = 1:16
    fprintf('Class %02.0f:\t\t\t\t\t%6.4f\n',i,R(i,13));
end
fprintf('\nPer-Class Specificity\n');
for i = 1:16
    fprintf('Class %02.0f:\t\t\t\t\t%6.4f\n',i,R(i,14));
end

fprintf('\n');
fprintf('Overall Accuracy:\t\t\t%6.4f\n',OA);
fprintf('Average Accuracy:\t\t\t%6.4f\n',AA);
fprintf('Average Precision:\t\t\t%6.4f\n',AP);
fprintf('Average Sensitivity:\t\t%6.4f\n',ASe);
fprintf('Average Specificity:\t\t%6.4f\n',ASp);
