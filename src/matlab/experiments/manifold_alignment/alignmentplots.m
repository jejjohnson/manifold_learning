function Stats = alignmentplots(Data, embedding, Options)

% get the number of components and test dimensions
nDomains = numel(Data);
% nComponents = Options.nComponents;



% get data
XTrain = [];
YTrain = [];

% grab appropriate training data
for idomain = 1:nDomains
    switch Options.algo
        case 'ssma'
            XTrain = [XTrain; embedding.ssma{idomain}.train];
            YTrain = [YTrain; Data{idomain}.Y.labeled];
            
        case 'sema'
            
            XTrain = [XTrain; embedding.sema{idomain}.train];
            YTrain = [YTrain; Data{idomain}.Y.labeled];
            
        case 'wang'
            
            XTrain = [XTrain; embedding.wang{idomain}.train];
            YTrain = [YTrain; Data{idomain}.Y.labeled];
            
    end
end


% Loop through data domains (HSIs)
for idomain = 1:nDomains
   
    
    % choose dimensions
    nDims = Options.nDims{idomain};
    


    XS = XTrain(:, 1:nDims);
    
    switch Options.algo
        case 'ssma'
            XTest = embedding.ssma{idomain}.test(:, 1:nDims);
            
        case 'sema'
            
            XTest = embedding.sema{idomain}.test(:, 1:nDims);
            
        case 'wang'
            
            XTest = embedding.wang{idomain}.test(:, 1:nDims);
            
    end
    YTest = Data{idomain}.YTest;

    switch lower(Options.method)

        % Linear Discriminant Analysis
        case 'lda'


            LDAObj = fitcdiscr(XS, YTrain);

            YPred = predict(LDAObj, XTest);

            % get statistics
            [~, Stats{idomain}] = classmetrics(YTest, YPred);

        % Support Vector Machines
        case 'svm'

            YPred = svmClassify(XS, YTrain, XTest);

            [~, Stats{idomain}] = classmetrics(YTest, YPred);

        otherwise
            error('Unrecognized classification method.');
    end
        
        


            


        

end
end

