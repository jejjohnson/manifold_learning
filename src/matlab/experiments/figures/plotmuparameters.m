function [varargout] = plotmuparameters(plotData, Options)


% Gather options
classMethod = Options.classMethod;
domain = Options.domain;
algorithm = Options.algo;
plotType = Options.plotType;
plotStat = Options.plotStat;
iCase = Options.iCase;
trainPrct = Options.trainPrct;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot
h = figure('Units', 'pixels', ...
    'Position', [100 100 500 400]);
h_line = surf(plotData);

h_line.LineStyle = '-.';
h_line.LineWidth = 2;
h_line.MeshStyle = 'row';
h_line.EdgeColor = 'interp';
h_line.FaceColor = 'texturemap';

% Choosing the Plot statistics
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
        
        switch algorithm
            case 'ssma'
            
                hTitle = title(sprintf('Case %d - SSMA', iCase));
            
            case 'wang'
                hTitle = title(sprintf('Case %d - Wang (Original)', iCase));
            
            case 'sema'
                hTitle = title(sprintf('Case %d - Spatial-Spectral Schroedinger', iCase));
            
           
        end
        hXlabel = xlabel('Embedding Dimensions');
        hZlabel = zlabel(plotStat);
        hYlabel = ylabel('\mu');
        
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
    
% Save Results
save_dest = 'experiments/figures/parameter_est/manifold_alignment/mu/';
save_file = char(sprintf('mu_%s_domain%d_%s_%s_train%d_case%d_%s_%s', ...
    dataset, domain, algorithm, plotType, trainPrct, iCase, classMethod, lower(plotStat)));
fn = char([save_dest, save_file]);
print(fn, '-depsc2');

% Output parameters
switch nargout
    case 0
        return;
    case 1
        varargout{1} = h;
    otherwise
        error('Incorrect number of outputs.');
end

end