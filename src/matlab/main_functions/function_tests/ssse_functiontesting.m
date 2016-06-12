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
clear options
options.type = 'standard';
options.saved = 1;
options.k = 20;
se_options.knn = options;
clear options

% Spatial knn graph construction options (Adjacency.m)
clear options
options.k = 4;
options.saved = 0;
se_options.spatial_nn = options;
clear options

% Eigenvalue Decompisition options (GraphEmbedding.m)
options.n_components = 150;
options.constraint = 'degree';
se_options.embedding = options;
clear options

% Schroedinger Eigenmaps options
se_options.type = 'spatialspectral';
se_options.image = img;


tic;
[embedding, lambda] = SchroedingerEigenmaps(imgVec, se_options);
time = toc;

% number of components we want to keep


fprintf('Schroedinger Eigenmaps: %.3f.\n', time)
% save('saved_data/ssse_eigvals.mat', 'embedding', 'lambda')


%#############################################################
%% Experiment II - SVM w/ Assessment versus dimension
%#############################################################

n_components = size(embedding,2);
test_dims = (1:10:n_components);

% choose training and testing amount
options.trainPrct = 0.75;
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
    embedding, gt_Vec, options);
    
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

hTitle = title('SSSE + LDA, SVM - Indian Pines');
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
print('saved_figures/ssse_ldasvm_test', '-depsc2');