function stats = alignmentclassification(Data, embedding, Options)

% get the number of components and test dimensions
nDomains = numel(Data);
nComponents = Options.nComponents;

% initialize domain for stats
stats = cell(1,nDomains);

% get data
XTrain = [];
YTrain = [];

% grab appropriate training data
for idomain = 1:nDomains
    XTrain = [XTrain; embedding{idomain}.train];
    YTrain = [YTrain; Data{idomain}.Y.labeled];
end


% Loop through data domains (HSIs)
for idomain = 1:nDomains

    % intialize statistics
    stats{idomain}.OA = []; 
    stats{idomain}.AA = []; 
    stats{idomain}.APr = []; 
    stats{idomain}.ASe = [];
    stats{idomain}.ASp = []; 
    stats{idomain}.k = []; 
    stats{idomain}.kv = [];
    stats{idomain}.dim = [];
    
    % get test dimensions
    testDims = size(Data{idomain}.XTrain, 2);
    
    % get step size
    if isfield(Options, 'dimStep')
        dimStep = Options.dimStep;
    else
        dimStep = 4;
    end
    
    % Loop through dimensions
    for iDim = 1:dimStep:testDims
        % number of dimensions
        
        XS = XTrain(:, 1:iDim);
        XTest = embedding{idomain}.test(:, 1:iDim);
        YTest = Data{idomain}.YTest;
        

        % Try Classification for dimension
        try
            
            % Choose Classification Method
            switch lower(Options.method)
                
                % Linear Discriminant Analysis
                case 'lda'


                    LDAObj = fitcdiscr(XS, YTrain);
                    
                    YPred = predict(LDAObj, XTest);

                    % get statistics
                    [~, newStats] = classmetrics(YTest, YPred);
                    
                % Support Vector Machines
                case 'svm'
                    
                    YPred = svmClassify(XS, YTrain, XTest);
                    
                    [~, newStats] = classmetrics(YTest, YPred);
                    
                otherwise
                    error('Unrecognized classification method.');
            end
            
            % save the relevant statistics
            stats{idomain}.OA = [stats{idomain}.OA; newStats.OA]; 
            stats{idomain}.AA = [stats{idomain}.AA; newStats.AA]; 
            stats{idomain}.APr = [stats{idomain}.APr; newStats.APr]; 
            stats{idomain}.ASe = [stats{idomain}.ASe; newStats.ASe];
            stats{idomain}.ASp = [stats{idomain}.ASp; newStats.ASp]; 
            stats{idomain}.k = [stats{idomain}.k; newStats.k]; 
            stats{idomain}.kv = [stats{idomain}.kv; newStats.v];
            stats{idomain}.dim = [stats{idomain}.dim; iDim];

        catch % leave NaN for failed classification attempts
            warning(sprintf('Failed at %d dimensions.', iDim));
            
            stats{idomain}.OA = [stats{idomain}.OA; NaN]; 
            stats{idomain}.AA = [stats{idomain}.AA; NaN]; 
            stats{idomain}.APr = [stats{idomain}.APr; NaN]; 
            stats{idomain}.ASe = [stats{idomain}.ASe; NaN];
            stats{idomain}.ASp = [stats{idomain}.ASp; NaN]; 
            stats{idomain}.k = [stats{idomain}.k; NaN]; 
            stats{idomain}.kv = [stats{idomain}.kv; NaN];
            stats{idomain}.dim = [stats{idomain}.dim; iDim];

        end
        

    end
end



end