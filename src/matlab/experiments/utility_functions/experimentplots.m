function [varargout] = experimentplots(lineData, options)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLOTTING WHICH EXPERIMENT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch lower(options.experiment)
    
    case 'statsdims'
        
        h = figure('Units', 'pixels', ...
            'Position', [100 100 500 375]);
        
        hold on;
        
        % plot line
        
        %==========================%
        % PLOTTING STATISTICS
        %==========================%
        
        hline = line(options.testDims, lineData);
        
        % set some first round of line parameters
        set(hLDA, ...
            'Color',        'r', ...
            'LineWidth',    2);
        set(hSVM, ...
            'Color',        'b', ...
            'LineWidth',    2);
        
        hXLabel = xlabel('d-Dimensions');
        hYLabel = ylabel('Kappa Coefficient');

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
            'XLim'      ,   [0 options.nComponents],...
            'YTick'     ,   0:0.1:1,...
            'XTick'     ,   0:10:options.nComponents,...
            'LineWidth' ,   1);
        
        % save the figure
        save_str = sprintf('%s_svm_%d', options.dimRed, options.nSamples);
        save_str = ['saved_figures/' save_str];
        print(save_str, '-depsc2');
        
    otherwise
        error([mfilename, 'experimentplots:badoptionexperimentinput'], ...
            'Error: invalid experiment chosen.');
end


% OUTPUTS %
switch nargout
    
    case 0
        return;
    case 1
        varargout{1} = h;
    otherwise
        error('Invalid number of output assignments.');
end

end
                    
                    
