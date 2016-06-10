% examplePartialKnowledgeScript.m: Provides example code for performing 
% spatial-spectral dimensionality reduction with partial knowledge of class
% labels, followed by SVM-based classification, as described in the paper:
%
% N. D. Cahill, S. E. Chew, and P. S. Wenger, "Spatial-Spectral 
% Dimensionality Reduction of Hyperspectral Imagery with Partial Knowledge
% of Class Labels," Proc. SPIE Defense & Security: Algorithms and 
% Technologies for Multispectral, Hyperspectral, and Ultraspectral Imagery
% XXI, April 2015. 
%
% This code requires the Spectral-Spatial Schroedinger Eigenmaps library,
% available at the MATLAB File Exchange, File No. 45908:
% http://www.mathworks.com/matlabcentral/fileexchange/45908-spatial-spectral-schroedinger-eigenmaps
%

%% Load Indian Pines data, available from:
% http://www.ehu.es/ccwintco/uploads/6/67/Indian_pines_corrected.mat
% http://www.ehu.es/ccwintco/uploads/6/67/Indian_pines_gt.mat
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

% get spatial positions of data
[x,y] = meshgrid(1:numCols,1:numRows);
pData = [x(:) y(:)];

%% display original image
imgColor = img(:,:,[29 15 12]);
iCMin = min(imgColor(:));
iCMax = max(imgColor(:));
imgColor = uint8(255*(imgColor - iCMin)./(iCMax-iCMin));
imgColorPad = padarray(imgColor,[1 1],0);

figure; imshow(imgColorPad); title('Indian Pines Image');

%% display ground truth and predicted label images
gtMask = gt>0;
gtClasses = uint8(255*ind2rgb(gt,hsv(16))).*repmat(uint8(gtMask),[1 1 3]) + ...
    255*repmat(uint8(~gtMask),[1 1 3]);
gtClassesPad = padarray(gtClasses,[1 1 0]);
figure; imshow(gtClassesPad); title('Ground Truth Class Labels');

alphaData = padarray(double(gtMask)*(3/4) + 1/4,[1 1],1);

%% read in manually provided class labels
pname = 'IndianPinesManualLabels\';
numClasses = 16;
Ccell = cell(numClasses,1);
C = false(numRows,numCols,numClasses);
for i = 1:numClasses
    Ccell{i} = imread([pname,sprintf('class%g.jpg',i)]);
    C(:,:,i) = any(Ccell{i}<225,3);
end

%% for each class, construct edges between each must-link (partial knowledge) constraint
constrInd = zeros(numRows*numCols,1);
MLcell = cell(numClasses,1);
for i = 1:numClasses
    ind = find(C(:,:,i));
    constrInd(ind) = i;
    MLcell{i} = completeGraph(ind);
end

%% specify the "active" set of classes to use ML (partial knowledge) constraints
% Note: specify MLactive to be a vector containing indices into the classes
% for which you would like to incorporate the provided partial knowledge.
% For example,
%   MLactive = 1:16;   % incorporates all classes
%   MLactive = 2;      % incorporates only corn-notill
%   MLactive = [2 5];  % incorporates corn-notill and grass-pasture 

MLactive = [2 5]; 

% concatenate ML constraints for all active classes
ML = cat(1,MLcell{MLactive});

%% display active constraints on IndianPines image
temp.R = imgColor(:,:,1);
temp.G = imgColor(:,:,2);
temp.B = imgColor(:,:,3);

cmap = hsv(16);
for i = 1:numel(MLactive)
    mask = C(:,:,MLactive(i));
    temp.R(mask) = uint8(255*cmap(MLactive(i),1));
    temp.G(mask) = uint8(255*cmap(MLactive(i),2));
    temp.B(mask) = uint8(255*cmap(MLactive(i),3));
end
imgColorMLCon = padarray(cat(3,temp.R,temp.G,temp.B),[1 1],0);

f = figure; imshow(imgColorMLCon); 
title('Indian Pines image with partial labels');

%% perform dimensionality reduction without and with ML constraints

% specify dimensionality reduction method. Can be one of: 'SM', 'GB',
% 'HZYZ', 'BE', 'BM', 'SSSE'
dimRedMethod = 'GB';

% specify whether or not to incorporate constraint conditioning
constraintConditioning = true;

% specify number of eigenvectors
numEigs = 50;

switch dimRedMethod
    case 'SM' % Shi-Malik
        
        % construct adjacency matrix
        rad = 5;
        sigmaF = 0.1;
        sigmaP = 100;
        fprintf('Constructing Shi-Malik Adjacency Matrix...\n');
        tic; A = adjacencyMatrixShiMalik(img,rad,sigmaF,sigmaP); toc
        
        % construct graph laplacian
        numNodes = size(A,1);
        D = spdiags(full(sum(A)).',0,numNodes,numNodes);
        L = D - A;
        
        % find first few dimensions of Laplacian eigenmap subspace
        fprintf('Computing Laplacian eigenmap for unconstrained Shi-Malik...\n');
        tic; [X.Unc,lambda.Unc] = schroedingerEigenmap(L,spalloc(numNodes,numNodes,0),0,numEigs); toc;

        % now include ML Constraints
        U = sparse(repmat((1:size(ML,1))',[1 2]),ML,...
            [ones(size(ML,1),1),-ones(size(ML,1),1)],...
            size(ML,1),numRows*numCols);

        % perform constraint conditioning if desired
        if constraintConditioning
            Dinv = spdiags(1./sum(A,2),0,size(A,1),size(A,1));
            U = U*Dinv*A;
            gamma = 0.01;
        else
            gamma = 0.1;
        end
        
        % construct cluster potential matrix
        UpU = U'*U;
        scUEqL = trace(L)./trace(UpU);
        
        % find first few dimensions of Laplacian eigenmap subspace
        fprintf('Computing Laplacian eigenmap for ML constrained Shi-Malik...\n');
        tic; [X.ML,lambda.ML] = schroedingerEigenmap(L,UpU,gamma*scUEqL,numEigs); toc
        
    case 'GB' % Gillis Bowles
        
        % construct adjacency matrix
        rad = 5;
        sigmaF = 0.2;
        sigmaP = 100;
        fprintf('Constructing Gillis-Bowles Adjacency Matrix...\n');
        tic; A = adjacencyMatrixGillisBowles(img,rad,sigmaF,sigmaP); toc
        
        % construct graph laplacian
        numNodes = size(A,1);
        D = spdiags(full(sum(A)).',0,numNodes,numNodes);
        L = D - A;
        
        % find first few dimensions of Laplacian eigenmap subspace
        fprintf('Computing Laplacian eigenmap for unconstrained Gillis-Bowles...\n');
        tic; [X.Unc,lambda.Unc] = schroedingerEigenmap(L,spalloc(numNodes,numNodes,0),0,numEigs); toc;

        % now include ML Constraints
        U = sparse(repmat((1:size(ML,1))',[1 2]),ML,...
            [ones(size(ML,1),1),-ones(size(ML,1),1)],...
            size(ML,1),numRows*numCols);

        % perform constraint conditioning if desired
        if constraintConditioning
            Dinv = spdiags(1./sum(A,2),0,size(A,1),size(A,1));
            U = U*Dinv*A;
            gamma = 0.01;
        else
            gamma = 0.1;
        end
        
        % construct cluster potential matrix
        UpU = U'*U;
        scUEqL = trace(L)./trace(UpU);
        
        % find first few dimensions of Laplacian eigenmap subspace
        fprintf('Computing Laplacian eigenmap for ML constrained Gillis-Bowles...\n');
        tic; [X.ML,lambda.ML] = schroedingerEigenmap(L,UpU,gamma*scUEqL,numEigs); toc

    case 'HZYZ'
        
        % construct adjacency matrix
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % % this code will do the construction, however it is slow.
        % sigmaF = 1;
        % sigmaP = 10;
        % k = 20;
        % [x,y] = meshgrid(1:numCols,1:numRows);
        % fprintf('Constructing HZYZ Adjacency Matrix...\n');
        % tic; A = adjacencyMatrixHZYZ(imgVec,[x(:),y(:)],k,sigmaF,sigmaP); toc
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % % to save time, load in the precomputed adajency matrix
        load('ExampleAdjacencyMatrices\Indian_pines_HZYZ_data.mat');
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % construct graph laplacian
        numNodes = size(A,1);
        D = spdiags(full(sum(A)).',0,numNodes,numNodes);
        L = D - A;
        
        % find first few dimensions of Laplacian eigenmap subspace
        fprintf('Computing Laplacian eigenmap for unconstrained HZYZ...\n');
        tic; [X.Unc,lambda.Unc] = schroedingerEigenmap(L,spalloc(numNodes,numNodes,0),0,numEigs); toc;

        % now include ML Constraints
        U = sparse(repmat((1:size(ML,1))',[1 2]),ML,...
            [ones(size(ML,1),1),-ones(size(ML,1),1)],...
            size(ML,1),numRows*numCols);

        % perform constraint conditioning if desired
        if constraintConditioning
            Dinv = spdiags(1./sum(A,2),0,size(A,1),size(A,1));
            U = U*Dinv*A;
            gamma = 0.01;
        else
            gamma = 0.1;
        end
        
        % construct cluster potential matrix
        UpU = U'*U;
        scUEqL = trace(L)./trace(UpU);
        
        % find first few dimensions of Laplacian eigenmap subspace
        fprintf('Computing Laplacian eigenmap for ML constrained HZYZ...\n');
        tic; [X.ML,lambda.ML] = schroedingerEigenmap(L,UpU,gamma*scUEqL,numEigs); toc

    case 'BE'
        
        % construct adjacency matrix
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % % this code will do the construction, however it is slow.
        % sigmaF = 1;
        % sigmaP = 10;
        % k = 20;
        %
        % [x,y] = meshgrid(1:numCols,1:numRows);
        % fprintf('Constructing Benedetto-E Adjacency Matrices...\n');
        % tic; A0 = adjacencyMatrixBenedetto(imgVec,[x(:) y(:)],k,sigmaF,sigmaP,0); toc
        % tic; A1 = adjacencyMatrixBenedetto(imgVec,[x(:) y(:)],k,sigmaF,sigmaP,1); toc
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % % to save time, load in the precomputed adajency matrix
        load('ExampleAdjacencyMatrices\Indian_pines_BE_data.mat');
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % construct graph laplacians
        numNodes = size(A0,1);
        D0 = spdiags(full(sum(A0)).',0,numNodes,numNodes);
        D1 = spdiags(full(sum(A1)).',0,numNodes,numNodes);
        L0 = D0 - A0;
        L1 = D1 - A1;
        
        % find first few dimensions of Laplacian eigenmap subspaces
        fprintf('Computing Laplacian eigenmap for beta = 0...\n');
        tic; [XS0,lambdaS0] = schroedingerEigenmap(L0,spalloc(numNodes,numNodes,0),0,numEigs+1); toc;
        
        fprintf('Computing Laplacian eigenmap for beta = 1...\n');
        tic; [XS1,lambdaS1] = schroedingerEigenmap(L1,spalloc(numNodes,numNodes,0),0,numEigs+1); toc;
        
        % fuse eigenvectors
        alpha = 42;
        X.Unc = [XS0(:,2:(1+alpha)),XS1(:,2:(numEigs-alpha+1))];
        
        % now include ML Constraints
        U = sparse(repmat((1:size(ML,1))',[1 2]),ML,...
            [ones(size(ML,1),1),-ones(size(ML,1),1)],...
            size(ML,1),numRows*numCols);

        % perform constraint conditioning if desired
        if constraintConditioning
            Dinv = spdiags(1./sum(A,2),0,size(A,1),size(A,1));
            U = U*Dinv*A;
            gamma = 0.01;
        else
            gamma = 0.1;
        end
        
        % construct cluster potential matrix
        UpU = U'*U;
        scUEqL0 = trace(L0)./trace(UpU);
        scUEqL1 = trace(L1)./trace(UpU);
        
        % find first few dimensions of Laplacian eigenmap subspace
        fprintf('Computing Laplacian eigenmap for ML constrained Benedetto-E with beta = 0...\n');
        tic; [XS0,lambdaS0] = schroedingerEigenmap(L0,UpU,gamma*scUEqL0,numEigs); toc
        fprintf('Computing Laplacian eigenmap for ML constrained Benedetto-E with beta = 1...\n');
        tic; [XS1,lambdaS1] = schroedingerEigenmap(L1,UpU,gamma*scUEqL1,numEigs); toc

        % fuse eigenvectors
        alpha = 42;
        X.ML = [XS0(:,2:(1+alpha)),XS1(:,2:(numEigs-alpha+1))];

    case 'BM'
        
        % construct adjacency matrix
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % % this code will do the construction, however it is slow.
        % sigmaF = 1;
        % sigmaP = 10;
        % k = 20;
        % alpha = 0.98;
        % [x,y] = meshgrid(1:numCols,1:numRows);
        % fprintf('Constructing Benedetto-M Adjacency Matrix...\n');
        % tic; A = adjacencyMatrixBenedetto(imgVec,[x(:) y(:)],k,sigmaF,sigmaP,alpha); toc
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % % to save time, load in the precomputed adajency matrix
        load('ExampleAdjacencyMatrices\Indian_pines_BM_data.mat');
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % construct graph laplacian
        numNodes = size(A,1);
        D = spdiags(full(sum(A)).',0,numNodes,numNodes);
        L = D - A;
        
        % find first few dimensions of Laplacian eigenmap subspace
        fprintf('Computing Laplacian eigenmap for unconstrained BM...\n');
        tic; [X.Unc,lambda.Unc] = schroedingerEigenmap(L,spalloc(numNodes,numNodes,0),0,numEigs); toc;

        % now include ML Constraints
        U = sparse(repmat((1:size(ML,1))',[1 2]),ML,...
            [ones(size(ML,1),1),-ones(size(ML,1),1)],...
            size(ML,1),numRows*numCols);

        % perform constraint conditioning if desired
        if constraintConditioning
            Dinv = spdiags(1./sum(A,2),0,size(A,1),size(A,1));
            U = U*Dinv*A;
            gamma = 0.01;
        else
            gamma = 0.1;
        end
        
        % construct cluster potential matrix
        UpU = U'*U;
        scUEqL = trace(L)./trace(UpU);
        
        % find first few dimensions of Laplacian eigenmap subspace
        fprintf('Computing Laplacian eigenmap for ML constrained BM...\n');
        tic; [X.ML,lambda.ML] = schroedingerEigenmap(L,UpU,gamma*scUEqL,numEigs); toc
        
    case 'SSSE'
        
        % select SSSE method you'd like to use
        % options:
        %   'SSSE_(SM)^(f,p)' (this is SSSE1 from paper 1)
        %   'SSSE_(SM)^(p,f)'
        %   'SSSE_(GB)^(f,p)' (this is SSSE2 from paper 1)
        %   'SSSE_(GB)^(p,f)'
        
        SSSEMethod = 'SSSE_(GB)^(f,p)';
        
        % parameters
        sigma = 0.2;
        k = 20;
        eta = 5;
        
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
                error('Invalid SSSE method.');
        end
        
        % construct graph laplacian
        numNodes = size(A,1);
        D = spdiags(full(sum(A)).',0,numNodes,numNodes);
        L = D - A;
        
        % create potential matrix to incorporate spatial connectivity
        V = schroedingerPotential(EData,CData,true,[1 1],idx);
        
        % determine scale factor that makes the potential matrix have the same
        % trace as L
        scVEqL = trace(L)./trace(V);
        
        % choose value of alpha to trade off L and V
        alpha = 17.78;
        
        % find first few dimensions of Schroedinger eigenmap subspace
        fprintf('Computing Schroedinger eigenmap for unconstrained SSSE...\n');
        tic; [X.Unc,lambda.Unc] = schroedingerEigenmap(L,V,alpha*scVEqL,numEigs); toc
                
        % now include ML Constraints
        U = sparse(repmat((1:size(ML,1))',[1 2]),ML,...
            [ones(size(ML,1),1),-ones(size(ML,1),1)],...
            size(ML,1),numRows*numCols);

        % perform constraint conditioning if desired
        if constraintConditioning
            Dinv = spdiags(1./sum(A,2),0,size(A,1),size(A,1));
            U = U*Dinv*A;
            gamma = 0.0001*alpha;
        else
            gamma = alpha;
        end
        
        % construct cluster potential matrix
        UpU = U'*U;
        scUEqL = trace(L)./trace(UpU);
        
        % find first few dimensions of Schroedinger eigenmap subspace
        fprintf('Computing Schroedinger eigenmap for ML constrained SSSE...\n');
        %tic;[X.ML,lambda.ML] = schroedingerEigenmap(L+alpha*scVEqL*V,UpU,gamma*scUEqL,numEigs); toc
        tic;[X.ML,lambda.ML] = schroedingerEigenmap(L,alpha*scVEqL*V+gamma*scUEqL*UpU,1,numEigs); toc

    otherwise
        error('Dimensionality reduction method not recognized.');
end

%% create training and testing data sets
trainPrct = 0.10;
rng('default'); % so each script generates the same training/testing data
[trainMask,testMask,gtMask] = createTrainTestData(gt,trainPrct);
trainInd = find(trainMask);

%% predict labels using SVM classifier
rng('default');
fprintf('Predicting class labels for unconstrained dimensionality reduction...\n');
tic; labels.Unc = svmClassify(X.Unc(trainInd,2:end),gt(trainInd),X.Unc(:,2:end)); toc

rng('default');
fprintf('Predicting class labels for ML constrained dimensionality reduction...\n');
tic; labels.ML = svmClassify(X.ML(trainInd,2:end),gt(trainInd),X.ML(:,2:end)); toc

%% display resulting class labels
% unconstrained
labelImg.Unc = reshape(labels.Unc,numRows,numCols);
imgClasses.Unc = uint8(255*ind2rgb(labelImg.Unc,hsv(16)));
imgClassesPad.Unc = padarray(imgClasses.Unc,[1 1],0);
figure; h = imshow(imgClassesPad.Unc); set(h,'alphaData',alphaData);
title('Unconstrained / SVM');

% ML constrained
labelImg.ML = reshape(labels.ML,numRows,numCols);
imgClasses.ML = uint8(255*ind2rgb(labelImg.ML,hsv(16)));
imgClassesPad.ML = padarray(imgClasses.ML,[1 1],0);
figure; h = imshow(imgClassesPad.ML); set(h,'alphaData',alphaData);
title('ML constrained / SVM');

%% compute accuracy measures
% unconstrained
CMat.Unc = confusionmat(gt(testMask&gtMask),labels.Unc(testMask&gtMask));
[BC.Unc,R.Unc] = binaryClassificationResults(CMat.Unc);
OA.Unc = trace(CMat.Unc)/sum(CMat.Unc(:));
AA.Unc = nanmean(R.Unc(:,11));
APr.Unc = nanmean(R.Unc(:,12));
ASe.Unc = nanmean(R.Unc(:,13));
ASp.Unc = nanmean(R.Unc(:,14));

% ML constrained
CMat.ML = confusionmat(gt(testMask&gtMask),labels.ML(testMask&gtMask));
[BC.ML,R.ML] = binaryClassificationResults(CMat.ML);
OA.ML = trace(CMat.ML)/sum(CMat.ML(:));
AA.ML = nanmean(R.ML(:,11));
APr.ML = nanmean(R.ML(:,12));
ASe.ML = nanmean(R.ML(:,13));
ASp.ML = nanmean(R.ML(:,14));

%% display results
fprintf('Per-Class Accuracy\t\t\tUnc\t\t\tML\n');
for i = 1:16
    fprintf('Class %02.0f:\t\t\t\t\t%6.4f\t\t%6.4f\n',i,R.Unc(i,11),R.ML(i,11));
end
fprintf('\nPer-Class Precision\t\t\tUnc\t\t\tML\n');
for i = 1:16
    fprintf('Class %02.0f:\t\t\t\t\t%6.4f\t\t%6.4f\n',i,R.Unc(i,12),R.ML(i,12));
end
fprintf('\nPer-Class Sensitivity\t\tUnc\t\t\tML\n');
for i = 1:16
    fprintf('Class %02.0f:\t\t\t\t\t%6.4f\t\t%6.4f\n',i,R.Unc(i,13),R.ML(i,13));
end
fprintf('\nPer-Class Specificity\t\tUnc\t\t\tML\n');
for i = 1:16
    fprintf('Class %02.0f:\t\t\t\t\t%6.4f\t\t%6.4f\n',i,R.Unc(i,14),R.ML(i,14));
end

fprintf('\n\t\t\t\t\t\t\tUnc\t\t\tML\n');
fprintf('Overall Accuracy:\t\t\t%6.4f\t\t%6.4f\n',OA.Unc,OA.ML);
fprintf('Average Accuracy:\t\t\t%6.4f\t\t%6.4f\n',AA.Unc,AA.ML);
fprintf('Average Precision:\t\t\t%6.4f\t\t%6.4f\n',APr.Unc,APr.ML);
fprintf('Average Sensitivity:\t\t%6.4f\t\t%6.4f\n',ASe.Unc,ASe.ML);
fprintf('Average Specificity:\t\t%6.4f\t\t%6.4f\n',ASp.Unc,ASp.ML);
