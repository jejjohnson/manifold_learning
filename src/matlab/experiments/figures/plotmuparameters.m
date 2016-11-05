function [varargout] = plotmuparameters(...
    x, y, plotData, Options)


% Gather options
classMethod = Options.classMethod;
domain = Options.domain;
algo = Options.algo;
plotType = Options.plotType;
plotStat = Options.plotStat;
iCase = Options.iCase;
trainPrct = Options.trainPrct;
experiment = Options.experiment;
dataset = Options.dataset;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot

switch lower(experiment)
    
    case 'mu'
        
        h = figure('Units', 'pixels', ...
            'Position', [100 100 500 400]);
        h_line = surf(x, y, plotData);

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
        
        % difference in plot types
        switch plotType

            case 'thesis'
                
                hXlabel = xlabel('\mu');
                hYlabel = ylabel('Dimensions');
                hZlabel = zlabel(plotStat);
                

                set([hXlabel, hYlabel, hZlabel], 'FontName', 'AvantGarde');
                set([hXlabel, hYlabel, hZlabel], 'FontSize', 12);    

            case 'prez'

                if isequal(algo, 'wang')

                    hTitle = title('Wang Original Method');

                elseif isequal(algo, 'ssma')
                    hTitle = title('SSMA');
                    
                elseif isequal(algo, 'sema')
                    hTitle = title('SEMA');

                end
                
                hXlabel = xlabel('\mu');
                
                hYlabel = ylabel('Dimensions');
                hZlabel = zlabel(plotStat);
                

                set([hXlabel, hYlabel, hZlabel, hTitle], 'FontName', 'AvantGarde');
                set([hXlabel, hYlabel, hZlabel, hTitle], 'FontSize', 12);    

            case 'pub'
                
                hXlabel = xlabel('\mu');
                hYlabel = ylabel('Dimensions');
                hZlabel = zlabel(plotStat);
                

                set([hXlabel, hYlabel, hZlabel, hTitle], 'FontName', 'AvantGarde');
                set([hXlabel, hYlabel, hZlabel, hTitle], 'FontSize', 12);    

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
            'YLim',         [0, max(y(:))],...
            'XLim',         [0, max(x(:))], ...
            'ZLim',         [0 1]);
    
        % save results
        switch lower(plotType)
            
            case 'thesis'
                
                % Save Results
                save_dest = ['E:\cloud_drives\dropbox\Apps\ShareLaTeX'...
                    '\RIT Masters Thesis Template\figures\ch5\exp2\'];
                save_file = char(sprintf('%s_mu_%s_c%d_d%d_%s_%s', ...
                    dataset, algo, Options.nCase, domain, classMethod, lower(plotStat)));
                fn = char([save_dest, save_file]);
                print(fn, '-depsc2');
                
            case 'pub'
            case 'prez'
            otherwise
                error('Unrecognized plotType for saving.');
        end

% Output parameters
switch nargout
    case 0
        return;
    case 1
        varargout{1} = h;
    otherwise
        error('Incorrect number of outputs.');
end

    case 'alpha'
        
        h = figure('Units', 'pixels', ...
            'Position', [100 100 500 400]);
        h_line = surf(x, y, plotData);

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
        
        % difference in plot types
        switch plotType

            case 'thesis'
                
                hXlabel = xlabel('\alpha');
                hYlabel = ylabel('Dimensions');
                hZlabel = zlabel(plotStat);
                

                set([hXlabel, hYlabel, hZlabel], 'FontName', 'AvantGarde');
                set([hXlabel, hYlabel, hZlabel], 'FontSize', 12);    

            case 'prez'

                if isequal(algo, 'wang')

                    hTitle = title('Wang Original Method');

                elseif isequal(algo, 'ssma')
                    hTitle = title('SSMA');
                    
                elseif isequal(algo, 'sema')
                    hTitle = title('SEMA');

                end
                
                hXlabel = xlabel('\alpha');
                hYlabel = ylabel('Dimensions');
                hZlabel = zlabel(plotStat);
                

                set([hXlabel, hYlabel, hZlabel, hTitle], 'FontName', 'AvantGarde');
                set([hXlabel, hYlabel, hZlabel, hTitle], 'FontSize', 12);    

            case 'pub'
                
                hXlabel = xlabel('\alpha');
                hYlabel = ylabel('Dimensions');
                hZlabel = zlabel(plotStat);
                

                set([hXlabel, hYlabel, hZlabel, hTitle], 'FontName', 'AvantGarde');
                set([hXlabel, hYlabel, hZlabel, hTitle], 'FontSize', 12);    

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
            'YLim',         [0 max(y(:))],...
            'XLim',         [0 max(x(:))], ...
            'ZLim',         [0 1]);
    
        % save results
        switch lower(plotType)
            
            case 'thesis'
                
                % Save Results
                save_dest = ['E:\cloud_drives\dropbox\Apps\ShareLaTeX\'...
                    'RIT Masters Thesis Template\figures\ch5\exp2\'];
                save_file = char(sprintf('%s_alpha_%s_c%d_d%d_%s_%s_%d', ...
                    dataset, algo, Options.nCase, domain, classMethod,...
                    lower(plotStat), Options.iMu));
                fn = char([save_dest, save_file]);
                print(fn, '-depsc2');
                
            case 'pub'
            case 'prez'
            otherwise
                error('Unrecognized plotType for saving.');
        end

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