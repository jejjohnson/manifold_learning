clear all; close all; clc;


% dataset
dataset = 'vcu';

% training amount
trainPrct = 1;

% get data
switch lower(dataset)
    
    case 'vcu'
        
        save_path = 'H:\Data\saved_data\manifold_alignment\parameter_estimation\';
        count = 1;
        for iCase = 1:4
            save_str = sprintf('%s_train%d_case%d',dataset, trainPrct, iCase);
            load([save_path save_str]);
            wang.SVM{count}= stats.wangsvm;
            wang.LDA{count} = stats.wanglda;
            ssma.SVM{count} = stats.ssmasvm;
            ssma.LDA{count} = stats.ssmalda;
            sema.SVM{count} = stats.semasvm;
            sema.LDA{count} = stats.semalda;
            count = count + 1;
        end
        
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot Options
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plotStat = 'OA';
plotType = 'prez';
classMethod = 'lda';
algo = 'wang';
domain = 1;
iCase = 1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Cases for Training Set 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

muValues = logspace(-1,2,10);
alphaValues = logspace(-1,2,10);
nMuValues = numel(muValues);
nAlphaValues = numel(alphaValues);
nDims{1} = 1:4:48;
nDims{2} = 1:4:96;

switch algo
    case {'ssma', 'wang'}
        switch domain
            case 1
                plotData = zeros(nMuValues, numel(nDims{1}));
            case 2
                plotData = zeros(nMuValues, numel(nDims{2}));
        end
    case {'sema'}
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


                case 'lda'

            end

    end
end


% Plot data
Options.algo = algo;
Options.plotStat = plotStat;
Options.domain = domain;
Options.classMethod = classMethod;
Options.plotType = plotType;
Options.iCase = iCase;
Options.trainPrct = trainPrct;

% Function
plotmuparameters(plotData, Options);
    


 