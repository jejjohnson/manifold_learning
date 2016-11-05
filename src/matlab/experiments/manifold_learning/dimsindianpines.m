% Classification Experiment - I
% 
%
% Data
% ----
% * Indian Pines HyperSpectral Image
% * Pavia Hyperspectral Image
%
% Algorithms
% ----------
% * Laplacian Eigenmaps
% * Locality Preserving Projections
% * Schroedinger Eigenmaps
% * Schroedinger Eigenmap Projections
%
% Output
% ------
% * OA      - Overall Accuracy
% * AA      - Average Accuracy
% * k       - Kappa Coefficient
% * AP      - Average Precision
% * Recall  - Recall
% * Dim     - Dimension
% * F1      - F1 Statistic
% * Time    - Time For Complete Algorithm (secs)
%
% Figures 
% -------
% * Classification Maps (all Algorithms)
% * k-NN Parameter Estimations
% * Potential Trade off Parameter Alpha
% * Potential Trade off Parameter Beta
% * % Samples vs. Dimensions
% * k-NN vs. Dimensions vs. % Samples
% * 
%
% Tables
% ------
% * Best Parameters Chosen
% * Statistical Significance
%
% Information
% -----------
% Author: Juan Emmanuel Johnson
% Email: jej2744@rit.edu
% Date: 27th July, 2016
%

% Clear Environment
clear all; close all; clc;

%============================%
% Gather Data for experiment %
%============================%

% add specific directory to matlab space
addpath('H:\Data\Images\RS\IndianPines\');      % specific to PC

img = importdata('Indian_pines_corrected.mat');
imgGT = importdata('Indian_pines_gt.mat');

% remove path from matlab space
rmpath('H:\Data\Images\RS\IndianPines\');  

%=========================================%
% Reorder and Rescale data into 2-D Array %
%=========================================%

[numRows, numCols, numSpectra] = size(img);
scfact = mean(reshape(sqrt(sum(img.^2,3)), numRows*numCols,1));
img = img./scfact;
imgVec = reshape(img, [numRows*numCols numSpectra]);
gtVec = reshape(imgGT, [numRows*numCols 1]);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Laplacian Eigenmaps
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%==================================%
% Spectral K-NN Graph Construction %
%==================================%

options = [];
options.type = 'standard';
options.saved = 0;
options.k = 20;
options.sigma = 1;

le_options.knn = options;                       % save options

%=================================%
% Eigenvalue Decompsition Options %
%=================================%

options = [];
options.n_components = 150;
options.constraint = 'degree';

le_options.embedding = options;                 % save options

%=================================%
% Laplacian Eigenmaps Algorithm %
%=================================%

tic;
le.embedding = LaplacianEigenmaps(imgVec, le_options);


%=================================%
% Classification Results
%=================================%

classOptions = [];
classOptions.nDims = 150;
classOptions.trainPrct = .10;
classOptions.experiment = 'statsdims';
classOptions.method = 'lda';
classOptions.gtVec = gtVec;
[Stats.le] = classexperiments(le.embedding, classOptions);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Schroedinger Eigenmaps
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%==================================%
% Spectral K-NN Graph Construction %
%==================================%

options = [];
options.type = 'standard';
options.saved = 0;
options.k = 20;
options.sigma = 1;

se_options.knn = options;                       % save options

%=================================%
% Spatial K-NN Graph Construction %
%=================================%

options = [];
options.k = 4;
options.saved = 0;

% save spatial knn options
se_options.spatial_nn = options;                % SE Algorithm


%=================================%
% Eigenvalue Decompsition Options %
%=================================%

options = [];
options.n_components = 150;
options.constraint = 'degree';

% save eigenvalue decomposition options
se_options.embedding = options;                % LPP Algorithm


%==========================%
% Spatial Spectral Options %
%==========================%

options = [];
options.image = img;

% save partial labels options
se_options.ss = options;                        % SE Algorithm

% schroedinger eigenmaps type
se_options.type = 'spaspec';                    % SE Algorithm

%========================%
% Schroedinger Eigenmaps %
%========================%

se.embedding = SchroedingerEigenmaps(imgVec, se_options);


%=================================%
% Classification Results
%=================================%

classOptions = [];
classOptions.nDims = 150;
classOptions.trainPrct = .10;
classOptions.experiment = 'statsdims';
classOptions.method = 'svm';
classOptions.gtVec = gtVec;
[Stats.se] = classexperiments(se.embedding, classOptions);

%% Locality Preserving Projections
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%==================================%
% Spectral K-NN Graph Construction %
%==================================%

options = [];
options.type = 'standard';
options.saved = 0;
options.k = 20;
options.sigma = 1;

lpp_options.knn = options;                       % save options

%=================================%
% Eigenvalue Decompsition Options %
%=================================%

options = [];
options.n_components = 150;
options.constraint = 'degree';

lpp_options.embedding = options;                 % save options

%=================================%
% Laplacian Eigenmaps Algorithm %
%=================================%


lpp.embedding = LocalityPreservingProjections(imgVec, lpp_options);
lpp.embedding = imgVec * lpp.embedding;


%=================================%
% Classification Results
%=================================%

classOptions = [];
classOptions.nDims = 150;
classOptions.trainPrct = .10;
classOptions.experiment = 'statsdims';
classOptions.method = 'lda';
classOptions.gtVec = gtVec;
[Stats.lpp] = classexperiments(lpp.embedding, classOptions);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Schroedinger Eigenmap Projections 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%==================================%
% Spectral K-NN Graph Construction %
%==================================%

options = [];
options.type = 'standard';
options.saved = 0;
options.k = 20;
options.sigma = 1;

sep_options.knn = options;                       % save options

%=================================%
% Spatial K-NN Graph Construction %
%=================================%

options = [];
options.k = 4;
options.saved = 0;

% save spatial knn options
sep_options.spatial_nn = options;                % sep Algorithm


%=================================%
% Eigenvalue Decompsition Options %
%=================================%

options = [];
options.n_components = 150;
options.constraint = 'degree';

% save eigenvalue decomposition options
sep_options.embedding = options;                % LPP Algorithm


%==========================%
% Spatial Spectral Options %
%==========================%

options = [];
options.image = img;

% save partial labels options
sep_options.ss = options;                        % sep Algorithm

% schroedinger eigenmaps type
sep_options.type = 'spaspec';                    % sep Algorithm

%========================%
% Schroedinger Eigenmaps %
%========================%



sep.embedding = SchroedingerEigenmapProjections(imgVec, sep_options);
sep.embedding = imgVec * sep.embedding;

%=================================%
% Classification Results
%=================================%

classOptions = [];
classOptions.nDims = 150;
classOptions.trainPrct = .10;
classOptions.experiment = 'statsdims';
classOptions.method = 'lda';
classOptions.gtVec = gtVec;
[Stats.sep] = classexperiments(sep.embedding, classOptions);




%------------------------------------------------
%% Plot
%-----------------------------------------------


testDims = 1:10:150;
figure('Units', 'pixels', ...
    'Position', [100 100 500 375]);
hold on;

% plot lines

hLE = line(testDims, Stats.le.k);
hSE = line(testDims, Stats.se.k);
hLPP = line(testDims, Stats.lpp.k(1:end-1,:));
hSEP = line(testDims, Stats.sep.k(1:end-1,:));

% set some first round of line parameters
set(hLE, ...
    'Color',        'r', ...
    'LineWidth',    2);
set(hSE, ...
    'Color',        'b', ...
    'LineWidth',    2);
set(hLPP, ...
    'Color',        'g', ...
    'LineWidth',    2);
set(hSEP, ...
    'Color',        'k', ...
    'LineWidth',    2);


hXLabel = xlabel('d-Dimensions');
hYLabel = ylabel('\kappa ');

hLegend = legend( ...
    [hLE, hSE, hLPP, hSEP], ...
    'LE',...
    'SE',...
    'LPP',...
    'SEP',...
    'location', 'SouthEast');

% pretty font and axis properties
set(gca, 'FontName', 'Helvetica');
set([hXLabel, hYLabel],...
    'FontName', 'AvantGarde');
set([hXLabel, hYLabel],...
    'FontSize',10);


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
    'XLim'      ,   [0 150],...
    'YTick'     ,   0:0.1:1,...
    'XTick'     ,   0:10:150,...
    'LineWidth' ,   1);

save_dest = ['E:\cloud_drives\dropbox\Apps\ShareLaTeX\'...
                    'RIT Masters Thesis Template\figures\ch5\exp1\'];
save_file = char(sprintf('indianpines_dims'));
fn = char([save_dest, save_file]);
print(fn, '-depsc2');