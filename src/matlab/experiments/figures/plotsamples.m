clear all; close all; clc;

% dataset
dataset = 'indianpines';

% Algorithm
algo = 'sep';

if isequal(dataset, 'indianpines')
    
    save_path = 'H:\Data\saved_data\samples_results\indianpines\';
    stats = cell(1,8);
    count = 1;
    for iSamples = 10:10:80
        save_data = sprintf('indianpines_%s_%d', algo, iSamples);
        load([save_path save_data]);
        switch algo
            case 'le'
                stats{count} = statsle;
            case 'se'
                stats{count} = statsse;
            case 'sep'
                stats{count} = statssep;
            case 'lpp'
                stats{count} = statslpp;
        end
        count = count + 1;
        
    end
    
elseif isequal(dataset, 'pavia')
    
    save_path = 'H:\Data\saved_data\samples_results\pavia\';
    stats = cell(1,8);
    count = 1;
    for iSamples = 10:10:80
        save_data = sprintf('pavia_%s_%d', algo, iSamples);
        load([save_path save_data]);
        switch algo
            case 'le'
                stats{count} = statsle;
            case 'se'
                stats{count} = statsse;
            case 'sep'
                stats{count} = statssep;
            case 'lpp'
                stats{count} = statslpp;
        end
        count = count + 1;
        
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot Options
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plotType = 'prez';       % pub, pres, paper

nExperiments = numel(stats);

alphaValues = 10:10:80;
plotStat = 'kappa';
nDimensions = 1:10:150;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Aggregate Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% get all lines into one surf
switch algo
    case {'lpp', 'sep'}
        plotData = zeros(numel(alphaValues), numel(nDimensions)+1);
    case {'le', 'se'}
        plotData = zeros(numel(alphaValues), numel(nDimensions));
end


for iAlpha = 1:numel(alphaValues)

        plotData(iAlpha, :) = getfield(stats{1, iAlpha}, plotStat);


end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

h = figure('Units', 'pixels', ...
    'Position', [100 100 500 400]);
h_line = surf(plotData);

h_line.LineStyle = '-.';
h_line.LineWidth = 2;
h_line.MeshStyle = 'row';
h_line.EdgeColor = 'interp';
h_line.FaceColor = 'texturemap';


% Choosing 
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
            
            hTitle = title('Number of Samples - Schroedinger Eigenmaps');
            
        elseif isequal(algo, 'sep')
            hTitle = title('Number of Samples - Schroedinger Eigenmap Projections');
        elseif isequal(algo, 'le')
            hTitle = title('Number of Samples - Laplacian Eignemaps');
        elseif isequal(algo, 'lpp')
             hTitle = title('Number of Samples - Locality Preserving Projections');
           
        end
        hXlabel = xlabel('Embedding Dimensions');
        hZlabel = zlabel(labelStat);
        hYlabel = ylabel('Samples');
        
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
save_file = char(sprintf('samples_%s_%s_%s_%s', dataset, algo, plotType, lower(plotStat)));
fn = char([save_dest, save_file]);
print(fn, '-depsc2');