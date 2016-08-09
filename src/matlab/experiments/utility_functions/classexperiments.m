function [varargout] = classexperiments(embedding, Options)
% options input
%   * nComponents
%   * trainPrct

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EXPERIMENT OPTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

switch lower(Options.experiment)
    
    case 'statsdims'
        
        nComponents = size(embedding, 2);
        testDims = (1:10:nComponents);
        
        % choose training and testing amount
        Options.trainPrct = 0.10;
        rng('default');     % reproducibility
        
%         h = waitbar(0, 'Initializing waitbar...');
        
        % initialize stats holding
        stats.OA = []; 
        stats.AA = []; 
        stats.APr = []; 
        stats.ASe = [];
        stats.ASp = []; 
        stats.k = []; 
        stats.kv = [];
        

        for iDim = testDims
            
            
            % # of dimensions
            XS = embedding(:,1:iDim);
            
            % training and testing samples
            [XTrain, YTrain, XTest, YTest] = ...
                traintestsplit(XS, Options.gtVec, Options);
            
            %========================%
            % CLASSIFICATION OPTIONS %
            %========================%
            switch lower(Options.method)

                case 'svm'      % Support Vector Machines (SVM)


%                     waitbar(iDim/nComponents,h,...
%                         'Performing SVM classifiaction')


                    % classifcaiton SVM
                    [YPred] = svmClassify(XTrain, YTrain, XTest);

                    [~, newStats] = classmetrics(YTest, YPred);

                case 'lda'      % Linear Discriminant Analysis (LDA)
                    
%                     waitbar(iDim/nComponents,h,...
%                         'Performing LDA classifiaction')

                    % classification LDA
                    LDAObj = fitcdiscr(XTrain, YTrain);

                    % get predictions
                    YPred = predict(LDAObj, XTest);

                    % get statistisc
                    [~, newStats] = classmetrics(YTest, YPred);

                otherwise
                    error([mfilename, ...
                        'classexperiments:badoptionsmethodinput'],...
                        'Error: unrecognized classification method.');
            end
            
            
            % save the relevant statistics
            stats.OA = [stats.OA; newStats.OA]; 
            stats.AA = [stats.AA; newStats.AA]; 
            stats.APr = [stats.APr; newStats.APr]; 
            stats.ASe = [stats.ASe; newStats.ASe];
            stats.ASp = [stats.ASp; newStats.ASp]; 
            stats.k = [stats.k; newStats.k]; 
            stats.kv = [stats.kv; newStats.v];
            
        end
        
%         close(h);
        
        % outputs
        switch nargout
            case 1
                varargout{1} = stats;
            otherwise
                error([mfilename, 'classexperiments:badvarargoutoutput'], ...
                    'Error: Invalid number of outputs.');
        end
        
    case 'classification'
        
         nComponents = size(embedding, 2);
        testDims = (1:10:nComponents);
        
        % choose training and testing amount
        Options.trainPrct = 0.10;
        rng('default');     % reproducibility
        
%         h = waitbar(0, 'Initializing waitbar...');
        
        % initialize stats holding
        stats.OA = []; 
        stats.AA = []; 
        stats.APr = []; 
        stats.ASe = [];
        stats.ASp = []; 
        stats.k = []; 
        stats.kv = [];
        

        for iDim = testDims
            
            
            % # of dimensions
            XS = embedding(:,1:iDim);
            
            % training and testing samples
            [XTrain, YTrain, XTest, YTest] = ...
                traintestsplit(XS, Options.gtVec, Options);
            
            %========================%
            % CLASSIFICATION OPTIONS %
            %========================%
            switch lower(Options.method)

                case 'svm'      % Support Vector Machines (SVM)


%                     waitbar(iDim/nComponents,h,...
%                         'Performing SVM classifiaction')


                    % classifcaiton SVM
                    [YPred] = svmClassify(XTrain, YTrain, XTest);

                    [~, newStats] = classmetrics(YTest, YPred);

                case 'lda'      % Linear Discriminant Analysis (LDA)
                    
%                     waitbar(iDim/nComponents,h,...
%                         'Performing LDA classifiaction')

                    % classification LDA
                    LDAObj = fitcdiscr(XTrain, YTrain);

                    % get predictions
                    YPred = predict(LDAObj, XTest);

                    % get statistisc
                    [~, newStats] = classmetrics(YTest, YPred);

                otherwise
                    error([mfilename, ...
                        'classexperiments:badoptionsmethodinput'],...
                        'Error: unrecognized classification method.');
            end
            
            
            % save the relevant statistics
            stats.OA = [stats.OA; newStats.OA]; 
            stats.AA = [stats.AA; newStats.AA]; 
            stats.APr = [stats.APr; newStats.APr]; 
            stats.ASe = [stats.ASe; newStats.ASe];
            stats.ASp = [stats.ASp; newStats.ASp]; 
            stats.k = [stats.k; newStats.k]; 
            stats.kv = [stats.kv; newStats.v];
            
        end
        
%         close(h);
        
        % outputs
        switch nargout
            case 1
                varargout{1} = stats;
            otherwise
                error([mfilename, 'classexperiments:badvarargoutoutput'], ...
                    'Error: Invalid number of outputs.');
        end
               
    case 'bestresults'
        
        % get dimensions
        nDims = Options.nDims;
        
        % get training and testing
        trainOptions.trainPrct = Options.trainPrct;
        [XTrain, YTrain, XTest, YTest] = traintestsplit(...
            embedding(:, 1:nDims),Options.gtVec, trainOptions);
        
        % get number of training samples
        nTraining = zeros(numel(unique(YTrain)),1);
        nTesting = zeros(numel(unique(YTest)),1);
        
        for iClass = 1:numel(unique(YTrain))
            nTraining(iClass) = numel(YTrain(YTrain==iClass)); 
            nTesting(iClass) = numel(YTest(YTest==iClass));
        end
        
        % Classification
        switch lower(Options.method)

            case 'svm'      % Support Vector Machines (SVM)


%                     waitbar(iDim/nComponents,h,...
%                         'Performing SVM classifiaction')


                % classifcaiton SVM
                [YPred] = svmClassify(XTrain, YTrain, XTest);

                [~, Stats, R] = classmetrics(YTest, YPred);
                Stats.R = R;

            case 'lda'      % Linear Discriminant Analysis (LDA)

%                     waitbar(iDim/nComponents,h,...
%                         'Performing LDA classifiaction')

                % classification LDA
                LDAObj = fitcdiscr(XTrain, YTrain);

                % get predictions
                YPred = predict(LDAObj, XTest);

                % get statistisc
                [~, Stats, R] = classmetrics(YTest, YPred);
                Stats.R = R;

            otherwise
                error([mfilename, ...
                    'classexperiments:badoptionsmethodinput'],...
                    'Error: unrecognized classification method.');
        end
        

        Stats.nDims = nDims;            % save dimensions
        Stats.nTraining = nTraining;    % save number of training
        Stats.nTesting = nTesting;      % save number of testing
        
        
        switch nargout
            case 1
                varargout{1} = Stats;
            otherwise
                error('Invalid Number of Outputs.');
        end

        
        
    case 'imagepredictions'
        
        % get dimensions
        nDims = Options.nDims;
        
        % get training and testing
        trainOptions.trainPrct = Options.trainPrct;
        [XTrain, YTrain, ~, ~] = traintestsplit(...
            embedding(:, 1:nDims),Options.gtVec, trainOptions);
        
        switch lower(Options.method)

            case 'svm'      % Support Vector Machines (SVM)


%                     waitbar(iDim/nComponents,h,...
%                         'Performing SVM classifiaction')


                % classifcaiton SVM
                [YPred] = svmClassify(XTrain, YTrain, embedding(:, 1:nDims));


            case 'lda'      % Linear Discriminant Analysis (LDA)

%                     waitbar(iDim/nComponents,h,...
%                         'Performing LDA classifiaction')

                % classification LDA
                LDAObj = fitcdiscr(XTrain, YTrain);

                % get predictions
                YPred = predict(LDAObj, embedding(:, 1:nDims));


            otherwise
                error([mfilename, ...
                    'classexperiments:badoptionsmethodinput'],...
                    'Error: unrecognized classification method.');
        end
        
        switch nargout
            case 1
                varargout{1} = YPred;
            otherwise
                error('Invalid number of outputs.');
        end

        
    otherwise
        error([mfilename, 'classexperiments:badoptionsexperimentinput'],...
                    'Error: unrecognized experiment.');
end


        

        

end