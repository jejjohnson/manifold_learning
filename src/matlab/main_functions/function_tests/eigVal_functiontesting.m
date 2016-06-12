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
%% Create Adjacency Matrix
%############################################

% Knn Graph Construction options (Adjacency.m)
clear options
options.type = 'standard';
options.saved = 1;
options.k = 20;

% create the adjacency matrix
[W, ~] = Adjacency(imgVec, options);
clear options

% Create diagonal degree matrix and laplacian matrix
D = spdiags(sum(W,2), 0, size(W,1), size(W,1));
L = D-W;

% create normalized laplacian (Jordan Weiss)
degs = sum(W,2);
D = sparse(1:size(W,1), 1:size(W,2), degs);
% avoid dividing by zero
degs(degs==0) = eps;
% calculate D^(-1/2)
D = spdiags(1./(degs.^0.5), 0, size(D,1), size(D,2));
% normalized Laplacian
L_norm = D * L * D;



%############################################
%% Try out some Eigenvalue Solvers
%############################################
n_components = 150;

decomp.matlab = 0;
decomp.jdqr = 0;
decomp.rsvd = 1;
decomp.ksvd = 0;
%------------------------------------------
% Matlabs - eigs function
%------------------------------------------

if decomp.matlab == 1
    
    disp('Computing Eigenvalues w/ MATLAB eigs...')
    tic;
    
    %normalized approach
    [embedding_mat, lambda_mat] = eigs(L_norm, ....
            round(1.5*(n_components+1)),'SM');
    % unnormalized approach
%     [embedding_mat, lambda_mat] = eigs(L, D,...
%             round(1.5*(n_components+1)),'SA');
    % discard smallest eigenvalue
    embedding_mat = embedding_mat(:, 2:n_components+1);
    lambda_mat = diag(lambda_mat); 
    lambda_mat=lambda_mat(2:n_components+1); 
    
    % plot the eigenvalues 
    figure;
    plot(lambda_mat);
    
    % print out the elapsed time
    elapsed_time = toc;
    fprintf('MATLAB - Eigenvalue Decomposition: %.3f.s\n', elapsed_time)
    disp('Completed...')
end


%------------------------------------------
% JDQR Function
%------------------------------------------
if decomp.jdqr == 1
    
    disp('Computing Eigenvalues w/ JDQR Function...')
    tic;
    

    warning('off')
    % calculate normalized Laplacian
    [embedding_jdqr, lambda_jdqr] = jdqr(L_norm,...
        round(1.75*(n_components+1)),'SM');
    warning('on')
    % discard smallest eigenvalue
%     embedding = embedding(:, 2:n_components+1);
%     lambda = diag(lambda); lambda=lambda(2:n_components+1); 
    lambda_jdqr = diag(lambda_jdqr);
    elapsed_time = toc;
    fprintf('JDQR - Eigenvalue Decomposition: %.3f.s\n', elapsed_time)
    disp('Completed...')
end

%------------------------------------------
% Randomized SVD
%------------------------------------------
if decomp.rsvd == 1
    
    disp('Computing Eigenvalues w/ Faster RSVD Function...')
    tic;
    
    % solve for the largest eigenvalue
    [~, Sigma_large, ~] = rsvd(L_norm, 1);
    
    % transform the original matrix A to A'
    L_new = 2.*Sigma_large.*eye(size(L_norm))-L_norm;
    
    % solve for the k largest components of the matrix A'
    [embedding_rsvd, Sigma, ~] = rsvd(L_new, n_components+1);
    
    % use the transformation to find the smallest eigenvalues
    lambda_rsvd = diag(2.*Sigma_large.*eye(n_components+1)- Sigma);
    

    % discard smallest eigenvalue
    embedding_rsvd = embedding_rsvd(:, 2:n_components+1);
    lambda_rsvd = diag(lambda_rsvd); 
    lambda_rsvd=lambda_rsvd(2:n_components+1); 
    
    % plot eigenvalues
    figure;
    plot(lambda_rsvd);
    elapsed_time = toc;
    fprintf('RSVD - Eigenvalue Decomposition: %.3f.s\n', elapsed_time)
    disp('Completed...')
end

%------------------------------------------
% Faster Randomized SVD
%------------------------------------------
if decomp.ksvd == 1
    
    disp('Computing Eigenvalues w/ Faster RSVD Function...')
    tic;
    
    % solve for the largest eigenvalue
    [~, Sigma_large, ~] = ksvdFaster(L_new, 1, 10, 10, 10);
    
    % transform the original matrix A to A'
    L_new = 2.*Sigma_large.*eye(size(L_norm))-L_norm;
    
    % solve for the k largest components of the matrix A'
    [embedding_ksvd, Sigma, ~] = ksvdFaster(L_new, ...
        n_components+1, n_components, n_components, n_components);
    
    % use the transformation to find the smallest eigenvalues
    lambda_ksvd = diag(2.*Sigma_large.*eye(n_components)- Sigma);
    

    % discard smallest eigenvalue
    embedding_ksvd = embedding_ksvd(:, 2:n_components);
    lambda_ksvd = diag(lambda_ksvd); 
    lambda_ksvd=lambda_ksvd(2:n_components); 
    
    % plot eigenvalues
    figure;
    plot(lambda_ksvd);
    elapsed_time = toc;
    fprintf('Faster RSVD - Eigenvalue Decomposition: %.3f.s\n', elapsed_time)
    disp('Completed...')
end

%#############################################################
%% Experiment I - Tuia et al. LDA & SVM w/ Assessment
%#############################################################
embedding = embedding_rsvd;

n_components = size(embedding,2);
test_dims = (1:10:n_components);
% testing and training

lda_class = [];
svm_class = [];

h = waitbar(0, 'Initializing waitbar...');

for dim = test_dims
    
    waitbar(dim/n_components,h, 'Performing LDA')
    % # of dimensions
    XS = embedding(:,1:dim);
    
    [Xtr, Ytr, Xts, Yts , ~, ~] = ppc(XS, gt_Vec, .01);
    
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

%% Plot

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

hTitle = title('LENorm + LDA, SVM; RSVD - Indian Pines');
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
    'YTick'     ,   0:0.1:1,...
    'XLim'      ,   [0 n_components],...
    'XTick'     ,   0:10:n_components,...
    'LineWidth' ,   1);

%% Save the figure
print('saved_figures/lenorm_rsvd_test', '-depsc2');
