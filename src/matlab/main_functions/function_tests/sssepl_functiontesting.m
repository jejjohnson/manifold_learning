%{ 
This is a sample script for the Schroedinger Eigenmaps algorithm.
I am going to walk through and deconstruct the script piece-by-piece.

Indian Pines Data Available from:
% http://www.ehu.es/ccwintco/uploads/6/67/Indian_pines_corrected.mat
% http://www.ehu.es/ccwintco/uploads/6/67/Indian_pines_gt.mat

%}
clear all; close all; clc;

%% Load Indian Pines Data

load('saved_data/img_data')

rng('default');     % reproducibility
%% Get some makeshift partial labels

tic;
X = imgVec; y = gt_Vec;

% extract 10 samples from each class
options = [];
options.trainPrnt = .10;
[~, ~, ~, ~, ~, masks] = train_test_split(X, y, options);

% set partial labels to be equal to the groundtruth
p_labels = y;

% let the test mask (unlabeled sample indices) be equal to 0
p_labels(masks.test) = 0;

% find the number of classes that are greater than 0
numClasses = numel(unique(y(y>0)));

% construct a cell to hold all of the partial labels
MLcell = cell(numClasses,1);

% loop through all class labels
for i = 1:numClasses
    
    % find the indices that have label i
    ind = find(p_labels(p_labels==i));
    
    % construct a complete graph w/ those indices
    MLcell{i} = completeGraph(ind);
    
end

% specify the active set of classes to use ML constraints
MLactive = 1:16;


% concatenate ML constraints for all active classes
ML = cat(1, MLcell{MLactive});

%% display original image
imgColor = img(:,:,[29 15 12]);
iCMin = min(imgColor(:));
iCMax = max(imgColor(:));
imgColor = uint8(255*(imgColor - iCMin)./(iCMax-iCMin));
imgColorPad = padarray(imgColor,[1 1],0);

figure; imshow(imgColorPad); title('Indian Pines Image');

%% display active constraints on IndianPines image

%% Construct adjacency matrix

options = [];
options.nn_graph = 'knn';
options.k = 20;
options.saved = 1;

[W, ~] = Adjacency(imgVec, options);

%% Construct Graph Laplacian
numNodes = size(W,1);
D = spdiags(full(sum(W)).', 0, numNodes, numNodes);
L = D - W;

%% Include ML Constraint's

U = sparse(repmat((1:size(ML,1))', [1 2]), ML, ...
    [ones(size(ML,1), 1), -ones(size(ML, 1), 1)], ...
    size(ML,1), size(img,1)*size(img,1));

%% Perform constrain conditioning
alpha = 17.78;
constraintConditioning = false;
if constraintConditioning
    Dinv = spdiags(1./sum(W,2), 0, size(W,1), size(W,1));
    U = U*Dinv*W;
    gamma = 1*alpha;
else
    gamma = alpha;
end

% partial label cluster potential
UpU = U'*U;

scUEqL = trace(L)./trace(UpU);

% Final matrix
A = L + gamma * scUEqL * UpU;
toc;
%% Find Eigenvalues and Eigenvectors
numEigs = 150;


tic;
[embedding, lambda] = eigs(A, D, numEigs, 'SM');
toc;



%#############################################################
%% Experiment II - SVM w/ Assessment versus dimension
%#############################################################

n_components = size(embedding,2);
test_dims = (1:10:n_components);

% choose training and testing amount
options = [];
options.trainPrct = 0.10;
rng('default');     % reproducibility

lda_OA = [];
svm_OA = [];

h = waitbar(0, 'Initializing waitbar...');

for dim = test_dims
    
    waitbar(dim/n_components,h, 'Performing SVM classification')
    % # of dimensions
    XS = embedding(:,1:dim);
    
    % training and testing samples
    [X_train, y_train, X_test, y_test] = train_test_split(...
    XS, gt_Vec, options);
    
    % classifcaiton SVM
    [y_pred] = svmClassify(X_train, y_train, X_test);
    
    [~, stats] = class_metrics(y_test, y_pred);
    
    svm_OA = [svm_OA; stats.OA];
    
    
    waitbar(dim/n_components,h, 'Performing LDA classification')
    % classifiaction LDA
    lda_obj = fitcdiscr(X_train, y_train);
    y_pred = predict(lda_obj, X_test);
    
    [~, stats] = class_metrics(y_test, y_pred);

    lda_OA = [lda_OA; stats.OA];
    

    
end

close(h)

%------------------------------------------------
% Plot
%-----------------------------------------------

figure('Units', 'pixels', ...
    'Position', [100 100 500 375]);
hold on;

% plot lines
hLDA = line(test_dims, lda_OA);
hSVM = line(test_dims, svm_OA);

% set some first round of line parameters
set(hLDA, ...
    'Color',        'r', ...
    'LineWidth',    2);
set(hSVM, ...
    'Color',        'b', ...
    'LineWidth',    2);

hTitle = title('SEPL + LDA, SVM - Indian Pines - 10');
hXLabel = xlabel('d-Dimensions');
hYLabel = ylabel('Correct Rate');

hLegend = legend( ...
    [hLDA, hSVM], ...
    'LDA - OA',...
    'SVM - OA',...
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
    'YLim'      ,   [0 1],...
    'XLim'      ,   [0 n_components],...
    'YTick'     ,   0:0.1:1,...
    'XTick'     ,   0:10:n_components,...
    'LineWidth' ,   1);

% Save the figure
print('saved_figures/sepl_ldasvm_test - 10', '-depsc2');

%% Best Results
rng('default'); % reproducibility

% training and testing
options.trainPrct = 0.25;
[X_train, y_train, X_test, y_test, idx, masks] = train_test_split(...
    embedding(:,1:50), gt_Vec, options);

% svm classificiton (w/ Image)
statoptions.imgVec = embedding(:,1:50);
[y_pred, imgClass] = svm_classify(X_train, y_train, X_test, statoptions);

masks.gt = reshape(masks.gt , [size(img,1) size(img,1)]);

% display ground truth and predicted label image
labelImg = reshape(imgClass,size(img,1),size(img,1));

figure;
subplot(2,2,1) ; imshow(gt,[0 max(gt(:))]); 
title('Ground Truth Class Labels');
subplot(2,2,2); imshow(masks.gt); 
title('Ground Truth Mask');
subplot(2,2,3); imshow(labelImg,[0 max(gt(:))]); 
title('Predicted Class Labels');
subplot(2,2,4); imshow(labelImg.*(masks.gt),[0 max(gt(:))]); 
title('Predicted Class Labels in Ground Truth Pixels');

% construct accuracy measures
[~, stats] = class_metrics(y_test, y_pred);

% display results
fprintf('\nCahills Experiment');
fprintf('\n\t\t\t\t\t\t\tSSSE\n');
fprintf('Kappa Coefficient:\t\t\t%6.4f\n',stats.k);
fprintf('Overall Accuracy:\t\t\t%6.4f\n',stats.OA);
fprintf('Average Accuracy:\t\t\t%6.4f\n',stats.AA);
fprintf('Average Precision:\t\t\t%6.4f\n',stats.APr);
fprintf('Average Sensitivity:\t\t%6.4f\n',stats.ASe);
fprintf('Average Specificity:\t\t%6.4f\n',stats.ASp);

%%

% %%
% % number of desired training datapoints per class
% num_training = 10;
% % ground truth vector
% y = gt_Vec;
% 
% % get a ground truth mask
% gtMask = (y~=0);
% 
% % unique groundtruth class labels
% uniqueLabels = unique(y(gtMask));
% 
% % loop through each class, selecting training data
% trainMask = false(size(y));
% for i = 1:numel(uniqueLabels)
%     
%     % get indices of points having current class label
%     currentClass = find(y==uniqueLabels(i));
%     
%     % randomly shuffle them
%     ccShuffled = currentClass(randperm(numel(currentClass)));
%     
%     % select a number for training
%     ccTrainInd = ccShuffled(1:num_training);
%     
%     % get training mask
%     trainMask(ccTrainInd) = true;
%     
% end
%         
% 
% % create test mask
% testMask = ~trainMask;
% 
% % get training and testing Indices
% trainInd = find(trainMask);
%     
%     






% %############################################
% %% SchroedingerEigenmaps function test
% %############################################
% 
% % Knn Graph Construction options (Adjacency.m)
% options = [];
% options.type = 'standard';
% options.saved = 1;
% options.k = 20;
% se_options.knn = options;
% 
% % Spatial knn graph construction options (Adjacency.m)
% options = [];
% options.k = 4;
% options.saved = 0;
% se_options.spatial_nn = options;
% 
% % Eigenvalue Decompisition options (GraphEmbedding.m)
% options = [];
% options.n_components = 150;
% options.constraint = 'identity';
% se_options.embedding = options;
% 
% 
% % Schroedinger Eigenmaps options
% se_options.type = 'partiallabels';
% se_options.image = img;
% se_options.alpha = 100;
% 
% [embedding, lambda] = SchroedingerEigenmapProjections(imgVec, se_options);
% 
% % project the data
% embedding = imgVec*embedding;
% 
% % fprintf('Schroedinger Eigenmaps: %.3f.\n', time)
% % save('saved_data/ssse_eigvals.mat', 'embedding', 'lambda')