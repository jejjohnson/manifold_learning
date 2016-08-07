clear all; close all; clc;

% dataset
dataset = 'indianpines';

% Algorithm
algo = 'se';

if isequal(dataset, 'indianpines')
    
    save_path = 'H:\Data\saved_data\alpha_results\indian_pines\';
    save_data = sprintf('%s_alpha_k20', algo);
    load([save_path save_data]);
elseif isequal(dataset, 'pavia')
    
    save_path = 'H:\Data\saved_data\alpha_results\pavia\';
    save_data = sprintf('%s_alpha_k20', algo);
    load([save_path save_data]);
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
plotStat = 'kappa';
nDimensions = 1:10:150;
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
% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

h = figure('Units', 'pixels', ...
    'Position', [100 100 500 400]);
h_line = surf(plotData);

h_line.LineStyle = '-.';
h_line.LineWidth = 2;
h_line.MeshStyle = 'row';
h_line.EdgeColor = 'interp';
h_line.FaceColor = 'texturemap';

switch plotStat 
    case {'kappa', 'k'}
        labelStat = '\kappa';
    case {'OA', 'OAccuracy'}
        labelStat = 'Overall Accuracy';
    case {'AA', 'AAccuracy'}
        labelStat = 'Average Accuracy';
    case {'APr', 'APrecision'}
        labelStat = 'Average Precision';
    case {'ASp', 'ASpecificity'}
        labelStat = 'Average Specificity';
    case {'ASe', 'ASensitivity'}
        labelStat = 'Average Sensitivity';
    case {'kv', 'v', 'kappaVariance'}
    otherwise
        error('Unrecognized Statistic');
end

% Difference in plots
switch plotType
    
    case 'pub'
        
    case 'prez'
        
        if isequal(algo, 'se')
            
            hTitle = title('\alpha Parameter - Schroedinger Eigenmaps');
            
        elseif isequal(algo, 'sep')
            hTitle = title('\alpha Parameter - Schroedinger Eigenmap Projections');

        end
        hXlabel = xlabel('Embedding Dimensions');
        hZlabel = zlabel('\kappa');
        hYlabel = ylabel('\alpha');
        
        set([hXlabel, hYlabel, hZlabel, hTitle], 'FontName', 'AvantGarde');
        set([hXlabel, hYlabel, hZlabel, hTitle], 'FontSize', 12);    
%         set(gca,...
%             'yTickLabel',   '',...
%             'xTickLabel',   '');
    case 'pub'
end


set(gca,...
    'Box',          'off',...
    'TickDir',      'out',...
    'XMinorTick',   'on',...
    'YMinorTick',   'on',...
    'YGrid',        'on',...
    'XColor',       [.3, .3, .3],...
    'YColor',       [.3, .3, .3],...
    'LineWidth',    1,...
    'YLim',         [0, 10],...
    'XLim',         [0 10], ...
    'ZLim',         [0 1]);
% 
% sef(gcf, 'PaperPositionMode', 'auto');

% Save Results
save_dest = 'experiments/figures/parameter_est/manifold_learning/';
save_file = char(sprintf('alphas_%s_%s_%s_%s', dataset, algo, plotType, lower(plotStat)));
fn = char([save_dest, save_file]);
print(fn, '-depsc2');