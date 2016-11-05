clear all; close all; clc;

% dataset
dataset = 'pavia';

% Algorithm
algo = 'se';

% classification method
classMethod = 'svm';

if isequal(dataset, 'indianpines')
    
    switch classMethod
        case 'svm'
            save_path = 'H:\Data\saved_data\alpha_results\indian_pines\';
            save_data = sprintf('%s_%s_alpha_%s_k20', algo, dataset, classMethod);
            load([save_path save_data]);
        case 'lda'
            save_path = 'H:\Data\saved_data\alpha_results\indian_pines\';
            save_data = sprintf('%s_%s_alpha_%s_k20', algo, dataset, classMethod);
            load([save_path save_data]);
        otherwise
            error('Unrecognized classification method for indian pines.');
    end
elseif isequal(dataset, 'pavia')
    switch classMethod
        case 'svm'    
            save_path = 'H:\Data\saved_data\alpha_results\pavia\';
            save_data = sprintf('%s_%s_alpha_%s_k20', algo, dataset, classMethod);
            load([save_path save_data]);
    
        case 'lda'
            save_path = 'H:\Data\saved_data\alpha_results\pavia\';
            save_data = sprintf('%s_%s_alpha_%s_k20', algo, dataset, classMethod);
            load([save_path save_data]);
        otherwise
            error('Unrecognized classification method for indian pines.');
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot Options
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plotType = 'prez';       % pub, pres, paper

if isequal(algo, 'sep')
    nExperiments = numel(statssep);
elseif isequal(algo, 'se')
    nExperiments = numel(statsse);
end

alphaValues = logspace(-1,2,10);
plotStat = 'k';
nDimensions = 1:10:100;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Aggregate Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% get all lines into one surf
if isequal(algo, 'se')
    plotData = zeros(numel(alphaValues), numel(nDimensions));
else
    plotData = zeros(numel(alphaValues), numel(nDimensions)+1);
end


for iAlpha = 1:numel(alphaValues)
    if isequal(algo, 'se')
        plotData(iAlpha, :) = getfield(statsse{1, iAlpha}, plotStat);
    else
        plotData(iAlpha, :) = getfield(statssep{1, iAlpha}, plotStat);

    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Options = [];
Options.classMethod = classMethod;
Options.experiment = 1;
Options.alg = algo;
Options.plotStat = 'k';
Options.plotType = 'thesis';
Options.dataset = dataset;


if isequal(algo, 'se')
    
    [x, y] = meshgrid(alphaValues, nDimensions);
else
    [x, y] = meshgrid(alphaValues, linspace(1,100,11));
end

plotalphaparameters(plotData', x, y, Options);


