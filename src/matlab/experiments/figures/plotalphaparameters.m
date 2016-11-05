function [varargout] = plotalphaparameters(...
    plotData, alphaValues, nDimensions, Options)

% Gather options
classMethod = Options.classMethod;
algorithm = Options.alg;
plotStat = Options.plotStat;
plotType = Options.plotType;
dataset = Options.dataset;

switch lower(Options.experiment)
    
    case 1
        % Plot
        h = figure('Units', 'pixels', ...
            'Position', [100 100 500 400]);
        h_line = surf(alphaValues, nDimensions, plotData);

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

            case 'thesis'
                
                hXlabel = xlabel('\alpha');
                hYlabel = ylabel('Dimensions');
                hZlabel = zlabel(plotStat);
                

                set([hXlabel, hYlabel, hZlabel], 'FontName', 'AvantGarde');
                set([hXlabel, hYlabel, hZlabel], 'FontSize', 12);    

            case 'prez'

                if isequal(algorithm, 'se')

                    hTitle = title('Schroedinger Eigenmaps');

                elseif isequal(algo, 'sep')
                    hTitle = title('Schroedinger Eigenmap Projections');

                end
                
                hXlabel = xlabel('\\alpha');
                hYlabel = ylabel('Dimensions');
                hZlabel = zlabel(plotStat);
                

                set([hXlabel, hYlabel, hZlabel, hTitle], 'FontName', 'AvantGarde');
                set([hXlabel, hYlabel, hZlabel, hTitle], 'FontSize', 12);    

            case 'pub'
                
                hXlabel = xlabel('\\alpha');
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
            'YLim',         [0 150],...
            'XLim',         [0 100], ...
            'ZLim',         [0 1]);
        
        
        % save results
        switch lower(plotType)
            
            case 'thesis'
                
                % Save Results
                save_dest = ['E:\cloud_drives\dropbox\Apps\ShareLaTeX\'...
                    'RIT Masters Thesis Template\figures\ch5\'];
                save_file = char(sprintf('%s_alpha_%s_%s', ...
                    dataset, algorithm, classMethod, lower(plotStat)));
                fn = char([save_dest, save_file]);
                print(fn, '-depsc2');
                
            case 'pub'
            case 'prez'
            otherwise
                error('Unrecognized plotType for saving.');
        end
    case 2
        
        domain = Options.domain;
        % Plot
        h = figure('Units', 'pixels', ...
            'Position', [100 100 500 400]);
        h_line = surf(alphaValues, nDimensions, plotData);

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

                if isequal(algorithm, 'ssma')

                    hTitle = title('SSMA');

                elseif isequal(algo, 'wang')
                    hTitle = title('Wang (Original)');

                elseif isequal(algo, 'sema')
                    hTitle = title('Spatial-Spectral Schroedinger');


                end
                hXlabel = xlabel('Embedding Dimensions');
                hZlabel = zlabel(plotStat);
                hYlabel = ylabel('\\alpha');

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
        save_dest = 'experiments/figures/parameter_est/manifold_alignment/alpha/';
        save_file = char(sprintf('alpha_%s_domain%d_%s_%s_%s', ...
            dataset, domain, plotType, classMethod, lower(plotStat)));
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
        
    otherwise
        error('Unrecognized experiment.');
end

end