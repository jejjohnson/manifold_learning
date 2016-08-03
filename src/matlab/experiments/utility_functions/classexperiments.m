function [varargout] = classexperiments(embedding, options)
% options input
%   * nComponents
%   * trainPrct

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EXPERIMENT OPTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

switch lower(options.experiment)
    
    case 'statsdims'
        
        nComponents = size(embedding, 2);
        testDims = (1:10:nComponents);
        
        % choose training and testing amount
        options.trainPrct = 0.10;
        rng('default');     % reproducibility
        
        h = waitbar(0, 'Initializing waitbar...');
        
        % initialize stats holding
        stats.OAccuracy = []; 
        stats.AAccuracy = []; 
        stats.APrecision = []; 
        stats.ASensitivity = [];
        stats.ASpecificity = []; 
        stats.kappa = []; 
        stats.kappaVariance = [];
        

        for iDim = testDims
            
            
            % # of dimensions
            XS = embedding(:,1:iDim);
            
            % training and testing samples
            [XTrain, YTrain, XTest, YTest] = ...
                traintestsplit(XS, options.gtVec, options);
            
            %========================%
            % CLASSIFICATION OPTIONS %
            %========================%
            switch lower(options.method)

                case 'svm'      % Support Vector Machines (SVM)


                    waitbar(iDim/nComponents,h,...
                        'Performing SVM classifiaction')


                    % classifcaiton SVM
                    [YPred] = svmClassify(XTrain, YTrain, XTest);

                    [~, newStats] = classmetrics(YTest, YPred);

                case 'lda'      % Linear Discriminant Analysis (LDA)
                    
                    waitbar(iDim/nComponents,h,...
                        'Performing LDA classifiaction')

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
            stats.OAccuracy = [stats.OAccuracy; newStats.OA]; 
            stats.AAccuracy = [stats.AAccuracy; newStats.AA]; 
            stats.APrecision = [stats.APrecision; newStats.APr]; 
            stats.ASensitivity = [stats.ASensitivity; newStats.ASe];
            stats.ASpecificity = [stats.ASpecificity; newStats.ASp]; 
            stats.kappa = [stats.kappa; newStats.k]; 
            stats.kappaVariance = [stats.kappaVariance; newStats.v];
            
        end
        
        close(h);
        
        % outputs
        switch nargout
            case 1
                varargout{1} = stats;
            otherwise
                error([mfilename, 'classexperiments:badvarargoutoutput'], ...
                    'Error: Invalid number of outputs.');
        end
        
    case 'bestresults'
        
        trainOptions = options.trainPrct;
        [XTrain, YTrain, XTest, YTest, idx, masks] = traintestsplit(...
            embedding(:, 1:50),options.gtVec, trainOptions);
        
        % SVM Classification (w/ Image)
        statoptions.imgVec = embedding(:, 1:50);
        [~, imgClassMap] = svm_classify(XTrain, YTrain, XTest, ...
            statoptions);
        % construct accuracy measures
        [~, stats] = classmetrics(yTest, yPred);
        fprintf('\n\t\t\t\t\t\t\tSSSE\n');
        fprintf('Kappa Coefficient:\t\t\t%6.4f\n',stats.k);
        fprintf('Overall Accuracy:\t\t\t%6.4f\n',stats.OA);
        fprintf('Average Accuracy:\t\t\t%6.4f\n',stats.AA);
        fprintf('Average Precision:\t\t\t%6.4f\n',stats.APr);
        fprintf('Average Sensitivity:\t\t%6.4f\n',stats.ASe);
        fprintf('Average Specificity:\t\t%6.4f\n',stats.ASp);
        
        
    case 'classmaps'
        
        trainOptions = options.trainPrct;
        [XTrain, YTrain, XTest, YTest, idx, masks] = traintestsplit(...
            embedding(:, 1:50),options.gtVec, trainOptions);
        
        % SVM Classification (w/ Image)
        statoptions.imgVec = embedding(:, 1:50);
        [~, imgClassMap] = svm_classify(XTrain, YTrain, XTest, ...
            statoptions);
        
        % 
        masks.gt = reshape(masks.gt, ...
            [size(options.img,1) size(options.img,1)]);
        
        % display ground truth and predicted label image
        labelImg = reshape(imgClassMap, ...
            size(option.img, 1), size(options.img,1));
        
        % plot ground truth image
        hGT = figure;   
        imshow(gt, [0 max(options.gt(:))]);
        
        % plot ground truth mask
        hGTMask = figure;
        imshow(masks.gt); 
        
        % Plot predicted class labels
        hClassLabels = figure;
        imshow(labelImg,[0 max(options.gt(:))]); 
        
        % Class Labels & Ground Truth Pixels
        hClassLabelsGT = figure;
        imshow(labelImg.*(masks.gt),[0 max(options.gt(:))]); 
        
        switch nargout
            case 0
                return;
            case 1
                imgFigures.hGT = hGT;
                imgFigures.hGT = hGTMask;
                imgFigures.ClassLabels = hClassLabels;
                imgFigures.hClassLabelsGT = hClassLabelsGT;
                varagout{1} = imgFigures;
            otherwise
                error([mfilename, 'classexperiments:badvarargoutoutput'], ...
                    'Error: Invalid number of outputs.');
        end
        
        
        
        
    otherwise
        error([mfilename, 'classexperiments:badoptionsexperimentinput'],...
                    'Error: unrecognized experiment.');
end


        

        

end