%{ 
This is a sample script for the Laplacian Eigenmaps algorithm.
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
options.k = 20;
options.saved = 2;
se_options.spatial_nn = options;
clear options

% Eigenvalue Decompisition options (GraphEmbedding.m)
options.n_components = 20;
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


tic;

fprintf('Schroedinger Eigenmaps: %.3f.\n', time)
save('saved_data/ssse_eigvals.mat', 'embedding', 'lambda')

%#############################################################
%% Experiment I - Tuia et al. LDA & SVM w/ Assessment
%#############################################################

n_components = size(embedding,2);
test_dims = (1:2:n_components);
% testing and training

lda_class = [];
svm_class = [];

h = waitbar(0, 'Initializing waitbar...');

for dim = test_dims
    
    waitbar(dim/n_components,h, 'Performing LDA')
    % # of dimensions
    XS = embedding(:,1:dim);
    
    [Xtr, Ytr, Xts, Yts , ~, ~] = ppc(XS, gt_Vec, .10);
    
    % Classifiaction - LDA
    [Ypred, err] = classify(Xts, Xtr, Ytr);
    
    % Assessment
    Results = assessment(Yts, Ypred, 'class');
    
    lda_class = [lda_class; (100-Results.OA)/100];
%     svm_class = [svm_class; Results.Kappa];
    
    waitbar(dim/n_components,h, 'Performing SVM')
    % Classification - SVM
    Ypred = svmClassify(Xtr, Ytr, Xts);
    
     % Assessment - SVM
    Results = assessment(Yts, Ypred, 'class');
    
    svm_class = [svm_class; (100-Results.OA)/100];
    
end

close(h)



figure('Units', 'pixels', ...
    'Position', [100 100 500 375]);
hold on;

% plot lines
hLDA = line(test_dims, lda_class);
hSVM = line(test_dims, svm_class);

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
    'YTick'     ,   0:0.1:1,...
    'XTick'     ,   0:10:100,...
    'LineWidth' ,   1);

%% Save the figure
print('saved_figures/ssse_test', '-depsc2');