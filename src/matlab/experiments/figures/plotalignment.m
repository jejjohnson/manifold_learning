clear all; close all; clc;


% dataset
dataset = 'vcu';
algo = 'lda';
nCase = 1;

% training amount
trainPrct = 1;

% get data
switch lower(dataset)
    
    case 'vcu'
        
        save_path = 'H:\Data\saved_data\manifold_alignment\parameter_estimation\';
        count = 1;
        for iCase = nCase
            save_str = sprintf('new2_%s_train%d_case%d',dataset, trainPrct, iCase);
            load([save_path save_str]);
            
            switch lower(algo)
                case 'lda'
                    wang.LDA{count} = stats.wanglda;
                    ssma.LDA{count} = stats.ssmalda;
                    sema.LDA{count} = stats.semalda;
                    
                case 'svm'
                    
                    wang.SVM{count}= stats.wangsvm;
                    ssma.SVM{count} = stats.ssmasvm;
                    sema.SVM{count} = stats.semasvm;
                    
                otherwise
                    error('unrecognized algorithm for dataset.')
            
            
            end
            count = count + 1;
        end
        
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot Options
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plotStat = 'k';
plotType = 'thesis';
classMethod = 'lda';
algo = 'sema';
domain = 2;
iCase = 1;
experiment = 'alpha';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Cases for Training Set 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

muValues = linspace(0,1,10);
alphaValues = logspace(-1,2,10);
nMuValues = numel(muValues);
nAlphaValues = numel(alphaValues);
nDims{1} = 1:4:48;
nDims{2} = 1:4:96;

switch algo
    case {'ssma', 'wang', 'sema'}
        switch domain
            case 1
                plotData = zeros(nMuValues, numel(nDims{1}));
            case 2
                plotData = zeros(nMuValues, numel(nDims{2}));
        end
end
    

% Plot parameters


% Choosing Plot Data
for iMu = 1:nMuValues
    switch lower(algo)
        case 'wang'             % Original Wang

            switch classMethod
                case 'svm'
                    
                    switch domain
                        case 1
                            plotData(iMu, :) = getfield(wang.SVM{1,iCase}{1,iMu}{1,1}, plotStat);
                        case 2
                            plotData(iMu, :) = getfield(wang.SVM{1,iCase}{1,iMu}{1,2}, plotStat);
                        otherwise
                            error('Invalid number of domains.')
                    end

                case 'lda'
                    switch domain
                        case 1
                            plotData(iMu, :) = getfield(wang.LDA{1,iCase}{1,iMu}{1,1}, plotStat);
                        case 2
                            plotData(iMu, :) = getfield(wang.LDA{1,iCase}{1,iMu}{1,2}, plotStat);
                        otherwise
                            error('Invalid number of domains.')
                    end
            end

        case 'ssma'         	% Tuia et al.
            switch classMethod
                case 'svm'
                    
                    switch domain
                        case 1
                            plotData(iMu, :) = getfield(ssma.SVM{1,iCase}{1,iMu}{1,1}, plotStat);
                        case 2
                            plotData(iMu, :) = getfield(ssma.SVM{1,iCase}{1,iMu}{1,2}, plotStat);
                        otherwise
                            error('Invalid number of domains.')
                    end

                case 'lda'
                    
                    switch domain
                        case 1
                            plotData(iMu, :) = getfield(ssma.LDA{1,iCase}{1,iMu}{1,1}, plotStat);
                        case 2
                            plotData(iMu, :) = getfield(ssma.LDA{1,iCase}{1,iMu}{1,2}, plotStat);
                        otherwise
                            error('Invalid number of domains.')
                    end
            end

        case 'sema'             % Schroedinger Eigenmaps

            switch classMethod
                case 'svm'
                    
                    for iAlpha = 1:nAlphaValues
                        
                        switch domain
                            case 1
                                plotData(iAlpha, :) = getfield(sema.SVM{1,iCase}{iAlpha,iMu}{1,1}, plotStat);
                                
                            case 2
                                plotData(iAlpha, :) = getfield(sema.SVM{1,iCase}{iAlpha,iMu}{1,2}, plotStat);
                            otherwise
                                error('Invalid number of domains.')
                        end
                    end
                    
                    AlphaPlots{iMu} = plotData;


                case 'lda'
                    
                    for iAlpha = 1:nAlphaValues
                        
                        switch domain
                            case 1
                                plotData(iAlpha, :) = getfield(sema.LDA{1,iCase}{iAlpha,iMu}{1,1}, plotStat);
                                
                            case 2
                                plotData(iAlpha, :) = getfield(sema.LDA{1,iCase}{iAlpha,iMu}{1,2}, plotStat);
                                
                            otherwise
                                error('Invalid number of domains.')
                        end
                    end
                    
                    AlphaPlots{iMu} = plotData;

            end

    end
end


%% Plot data
Options.algo = algo;
Options.plotStat = plotStat;
Options.domain = domain;
Options.classMethod = classMethod;
Options.plotType = plotType;
Options.iCase = iCase;
Options.trainPrct = trainPrct;
Options.experiment = experiment;
Options.dataset = dataset;
Options.nCase = nCase;



% Function

switch lower(algo)
    
    case {'ssma', 'wang'}
        [x, y] = meshgrid(muValues, nDims{domain});
        plotmuparameters(x, y, plotData', Options);
        
    case {'sema'}
        [x, y] = meshgrid(alphaValues, nDims{domain});
        for iMu = 1:nMuValues
            Options.iMu = iMu;
            plotmuparameters(x, y, AlphaPlots{iMu}', Options);

        end
        
    otherwise
        error('Unrecognized algo for plotting.');
end

    


 