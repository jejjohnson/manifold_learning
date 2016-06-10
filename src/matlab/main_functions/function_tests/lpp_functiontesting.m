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

%#########################################
%% Construct Adjacency Matrix - Test
%#########################################

% K NEAREST NEIGHBORS - standard spectral
clear options
options.type = 'standard';
options.saved = 1;
options.k = 20;

W = Adjacency(imgVec, options);


figure(1);
spy(W, 1e-15)

%############################################
%% LocalityPreservingProjections function test
%############################################

clear options
n_components = 20;
options.n_components = n_components;

tic;
[projection, lambda] = LocalityPreservingProjections(W, imgVec, options);
time = toc;



tic;

fprintf('Eigenvalue Decomposition: %.3f.\n', time)
save('saved_data/lpp_eigfunc.mat', 'embedding', 'lambda')

embedding = imgVec*projection;
%#############################################################
%% Experiment III - Tuia et al. LDA w/ Assessment
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

hTitle = title('LE + LDA - Indian Pines');
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
