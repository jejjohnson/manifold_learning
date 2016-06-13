%{ 
This is a sample script for the Locality Preserving Projections algorithm.
I am going to walk through and deconstruct the script piece-by-piece.

Indian Pines Data Available from:
% http://www.ehu.es/ccwintco/uploads/6/67/Indian_pines_corrected.mat
% http://www.ehu.es/ccwintco/uploads/6/67/Indian_pines_gt.mat

%}
clear all; close all; clc;

%% Load Indian Pines Data

try
    
    load('saved_data/img_data')
    disp('found previous image data')
catch
    % add file path of where my data is located
    addpath('H:\Data\Images\RS\IndianPines\')
    % addpath('/media/eman/Emans HDD/Data/Images/RS/IndianPines')

    % load the images
    load('Indian_pines_corrected.mat');
    load('Indian_pines_gt.mat');

    % set them to variables
    img = indian_pines_corrected;
    gt = indian_pines_gt;

    % clear the path as well as the datafiles 
    clear indian*
    % rmpath('/media/eman/Emans HDD/Data/Images/RS/IndianPines')
    rmpath('H:\Data\Images\RS\IndianPines\')
end


%############################################
%% LocalityPreservingProjections function test
%############################################

% Knn Graph Construction options (Adjacency.m)
options = [];
options.type = 'standard';
options.saved = 1;
options.k = 20;
lpp_options.knn = options;


% Eigenvalue Decompisition options (GraphEmbedding.m)
options = [];
options.n_components = 150;
options.constraint = 'degree';
lpp_options.embedding = options;



tic;
[embedding, lambda] = LocalityPreservingProjections(imgVec, lpp_options);
time = toc;

% number of components we want to keep

embedding = imgVec*embedding;
tic;

fprintf('Locality Preserving Projections: %.3f.\n', time)
% save('saved_data/le_eigvals.mat', 'embedding', 'lambda')

%#############################################################
%% Experiment II - SVM w/ Assessment versus dimension
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
    
    waitbar(dim/n_components,h, 'Performing SVM classifiaction')
    % # of dimensions
    XS = embedding(:,1:dim);
    
    % training and testing samples
    [X_train, y_train, X_test, y_test] = train_test_split(...
    XS, gt_Vec, options);
    
    % classifcaiton SVM
    [y_pred] = svmClassify(X_train, y_train, X_test);
    
    [~, stats] = class_metrics(y_test, y_pred);
    
    svm_OA = [svm_OA; stats.OA];
    
    
    waitbar(dim/n_components,h, 'Performing LDA classifiaction')
    % classifiaction LDA
    lda_obj = fitcdiscr(X_train, y_train);
    y_pred = predict(lda_obj, X_test);
    
    [~, stats] = class_metrics(y_test, y_pred);

    lda_OA = [lda_OA; stats.OA];
    

    
end

close(h)
% 
%% Plot

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

hTitle = title('LPP + LDA, SVM - Indian Pines - 10');
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

%% Save the figure
print('saved_figures/lpp_ldasvm_test_10', '-depsc2');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Experiment II
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

test_dims = (1:10:150);

rng('default');     % reproducibility

lda_OA = [];
svm_OA = [];

h = waitbar(0, 'Initializing waitbar...');

for dim = test_dims
    
    %----------------------------------
    % Training Vs. Testing
    %----------------------------------
    
    options = [];
    options.trainPrct = 0.10;
    
    [X_train, y_train, X_test, y_test] = train_test_split(...
    imgVec, gt_Vec, options);
    
    %----------------------------------
    % Dimensionality Reduction
    %----------------------------------
    
    waitbar(dim/n_components,h, 'Performing DR: LPP')
    
    % Knn Graph Construction options (Adjacency.m)
    options = [];
    options.type = 'standard';
    options.saved = 0;
    options.k = 20;
    lpp_options.knn = options;

    % Eigenvalue Decompisition options (GraphEmbedding.m)
    options = [];
    options.n_components = dim;
    options.constraint = 'degree';
    lpp_options.embedding = options;

    % find the embedding
    [embedding, lambda] = LocalityPreservingProjections(...
        X_train, lpp_options);
    


    % project the training data
    X_train_proj = X_train * embedding;
    
    % project the testing data
    X_test_proj = X_test * embedding;
    %--------------------------------
    % Classification
    %--------------------------------
    
    waitbar(dim/n_components,h, 'Performing SVM classifiaction')
   
    % classifcaiton SVM
    [y_pred] = svmClassify(X_train_proj, y_train, X_test_proj);
    
    % get classification metrics
    [~, stats] = class_metrics(y_test, y_pred);
    
    % save results
    svm_OA = [svm_OA; stats.OA];
    
    
    waitbar(dim/n_components,h, 'Performing LDA classifiaction')
    
    % classifiaction LDA
    lda_obj = fitcdiscr(X_train_proj, y_train);
    y_pred = predict(lda_obj, X_test_proj);
    
    % get classification metrics
    [~, stats] = class_metrics(y_test, y_pred);

    % save results
    lda_OA = [lda_OA; stats.OA];
    

    
end

close(h)
% 
% Plot

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

hTitle = title('LPP + LDA, SVM - v2 - Indian Pines - 10');
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
print('saved_figures/lpp_ldasvm_v2_test_10', '-depsc2');


