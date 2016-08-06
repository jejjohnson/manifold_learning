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

%############################################
%% SchroedingerEigenmaps function test
%############################################

% Knn Graph Construction options (Adjacency.m)
options = [];
options.type = 'standard';
options.saved = 1;
options.k = 20;
se_options.knn = options;

% Spatial knn graph construction options (Adjacency.m)
options = [];
options.k = 4;
options.saved = 0;
se_options.spatial_nn = options;

% Eigenvalue Decompisition options (GraphEmbedding.m)
options = [];
options.n_components = 150;
options.constraint = 'identity';
se_options.embedding = options;


% Schroedinger Eigenmaps options
se_options.type = 'spatialspectral';
se_options.image = img;
se_options.alpha = 100;

[embedding, lambda] = SchroedingerEigenmapProjections(imgVec, se_options);

% project the data
embedding = imgVec*embedding;

% fprintf('Schroedinger Eigenmaps: %.3f.\n', time)
% save('saved_data/ssse_eigvals.mat', 'embedding', 'lambda')


%#############################################################
%% Experiment I - SVM w/ Assessment versus dimension
%#############################################################

n_components = size(embedding,2);
test_dims = (1:10:n_components);

% choose training and testing amount
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
    [X_train, y_train, X_test, y_test] = traintestsplit(...
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

%-----------------------------------------------
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

hTitle = title('SSSEP + LDA, SVM - Indian Pines - 10');
hXLabel = xlabel('d-Dimensions');
hYLabel = ylabel('Correct Rate');

hLegend = legend( ...
    [hLDA, hSVM], ...
    'LDA - OA',...
    'SVM - OA',...
    'location', 'NorthEast');

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

%-----------------------------------------------
% Save the Figure
%-----------------------------------------------
print('saved_figures/sssep_ldasvm_test_10', '-depsc2');

% %% Best Results
% rng('default'); % reproducibility
% 
% % training and testing
% options.trainPrct = 0.01;
% [X_train, y_train, X_test, y_test, idx, masks] = train_test_split(...
%     embedding(:,1:50), gt_Vec, options);
% 
% % svm classificiton (w/ Image)
% statoptions.imgVec = embedding(:,1:50);
% [y_pred, imgClass] = svm_classify(X_train, y_train, X_test, statoptions);
% 
% masks.gtMask = reshape(masks.gtMask , [size(img,1) size(img,1)]);
% 
% % display ground truth and predicted label image
% labelImg = reshape(imgClass,size(img,1),size(img,1));
% 
% figure;
% subplot(2,2,1) ; imshow(gt,[0 max(gt(:))]); 
% title('Ground Truth Class Labels');
% subplot(2,2,2); imshow(masks.gtMask); 
% title('Ground Truth Mask');
% subplot(2,2,3); imshow(labelImg,[0 max(gt(:))]); 
% title('Predicted Class Labels');
% subplot(2,2,4); imshow(labelImg.*(masks.gtMask),[0 max(gt(:))]); 
% title('Predicted Class Labels in Ground Truth Pixels');
% 
% % construct accuracy measures
% [~, stats] = class_metrics(y_test, y_pred);
% 
% % display results
% fprintf('\nCahills Experiment');
% fprintf('\n\t\t\t\t\t\t\tSSSE\n');
% fprintf('Kappa Coefficient:\t\t\t%6.4f\n',stats.k);
% fprintf('Overall Accuracy:\t\t\t%6.4f\n',stats.OA);
% fprintf('Average Accuracy:\t\t\t%6.4f\n',stats.AA);
% fprintf('Average Precision:\t\t\t%6.4f\n',stats.APr);
% fprintf('Average Sensitivity:\t\t%6.4f\n',stats.ASe);
% fprintf('Average Specificity:\t\t%6.4f\n',stats.ASp);


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% Experiment I
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% test_dims = (1:10:150);
% 
% rng('default');     % reproducibility
% 
% lda_OA = [];
% svm_OA = [];
% 
% h = waitbar(0, 'Initializing waitbar...');
% 
% for dim = test_dims
%     
%     %----------------------------------
%     % Training Vs. Testing
%     %----------------------------------
%     
%     options = [];
%     options.trainPrct = 0.1;
%     
%     [X_train, y_train, X_test, y_test] = train_test_split(...
%     imgVec, gt_Vec, options);
%     
%     %----------------------------------
%     % Dimensionality Reduction
%     %----------------------------------
%     
%     waitbar(dim/n_components,h, 'Performing DR: SSSEP')
%     
%     % Knn Graph Construction options (Adjacency.m)
%     options = [];
%     options.type = 'standard';
%     options.saved = 0;
%     options.k = 20;
%     se_options.knn = options;
% 
%     % Spatial knn graph construction options (Adjacency.m)
%     options = [];
%     options.k = 4;
%     options.saved = 0;
%     se_options.spatial_nn = options;
% 
%     % Eigenvalue Decompisition options (GraphEmbedding.m)
%     options = [];
%     options.n_components = 150;
%     options.constraint = 'degree';
%     se_options.embedding = options;
% 
% 
%     % Schroedinger Eigenmaps options
%     se_options.type = 'spatialspectral';
%     se_options.image = img;
%     
%     % find the embedding
%     [embedding, lambda] = SchroedingerEigenmapProjections(...
%         X_train, se_options);
% 
%     % project the training data
%     X_train_proj = X_train * embedding;
%     
%     % project the testing data
%     X_test_proj = X_test * embedding;
%     
%     %--------------------------------
%     % Classification
%     %--------------------------------
%     
%     waitbar(dim/n_components,h, 'Performing SVM classifiaction')
%    
%     % classifcaiton SVM
%     [y_pred] = svmClassify(X_train_proj, y_train, X_test_proj);
%     
%     % get classification metrics
%     [~, stats] = class_metrics(y_test, y_pred);
%     
%     % save results
%     svm_OA = [svm_OA; stats.OA];
%     
%     
%     waitbar(dim/n_components,h, 'Performing LDA classifiaction')
%     
%     % classifiaction LDA
%     lda_obj = fitcdiscr(X_train_proj, y_train);
%     y_pred = predict(lda_obj, X_test_proj);
%     
%     % get classification metrics
%     [~, stats] = class_metrics(y_test, y_pred);
% 
%     % save results
%     lda_OA = [lda_OA; stats.OA];
%     
% 
%     
% end
% 
% close(h)
% 
% 
% %----------------
% % Plot
% %----------------
% 
% figure('Units', 'pixels', ...
%     'Position', [100 100 500 375]);
% hold on;
% 
% % plot lines
% hLDA = line(test_dims, lda_OA);
% hSVM = line(test_dims, svm_OA);
% 
% % set some first round of line parameters
% set(hLDA, ...
%     'Color',        'r', ...
%     'LineWidth',    2);
% set(hSVM, ...
%     'Color',        'b', ...
%     'LineWidth',    2);
% 
% hTitle = title('SSSEP + LDA, SVM - v2 - Indian Pines');
% hXLabel = xlabel('d-Dimensions');
% hYLabel = ylabel('Correct Rate');
% 
% hLegend = legend( ...
%     [hLDA, hSVM], ...
%     'LDA - OA',...
%     'SVM - OA',...
%     'location', 'NorthWest');
% 
% % pretty font and axis properties
% set(gca, 'FontName', 'Helvetica');
% set([hTitle, hXLabel, hYLabel],...
%     'FontName', 'AvantGarde');
% set([hXLabel, hYLabel],...
%     'FontSize',10);
% set(hTitle,...
%     'FontSize'  ,   12,...
%     'FontWeight',   'bold');
% 
% set(gca,...
%     'Box',      'off',...
%     'TickDir',  'out',...
%     'TickLength',   [.02, .02],...
%     'XMinorTick',   'on',...
%     'YMinorTick',   'on',...
%     'YGrid',        'on',...
%     'XColor',       [.3,.3,.3],...
%     'YColor',       [.3, .3, .3],...
%     'YLim'      ,   [0 1],...
%     'XLim'      ,   [0 n_components],...
%     'YTick'     ,   0:0.1:1,...
%     'XTick'     ,   0:10:n_components,...
%     'LineWidth' ,   1);
% 
% %----------------
% % Save the figure
% %----------------
% print('saved_figures/sssep_ldasvm_v2_test', '-depsc2');
