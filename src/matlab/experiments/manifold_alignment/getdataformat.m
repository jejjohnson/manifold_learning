function Data = getdataformat(ImageData, options)

% number of domains
nDomains = numel(ImageData);
Data = cell(1,nDomains);            % Initialize Data cell for storage


for idomain = 1:nDomains
    
    
    
    % get labeled and unlabeled data;
    TrainOptions = [];
    TrainOptions.trainPrct = options.trainPrct{idomain};
    
    [XTrain, YTrain, XTest, YTest] = ...
        traintestsplit(...
        ImageData{idomain}.imgVec, ...
        ImageData{idomain}.gtVec, ...
        TrainOptions);
    
    % get spatial locations of matrix
    [x, y] = meshgrid(1:ImageData{idomain}.dims.numCols, ...
        1:ImageData{idomain}.dims.numRows);
    pData = [x(:) y(:)];


    
    % get unlabeled training and unlabeled testing
    LabeledOptions = [];
    LabeledOptions.trainPrct = options.labelPrct{idomain};
    
    [XLabeled, YLabeled, XUnlabeled, YUnlabeled, IDX] = ...
        traintestsplit(XTrain, YTrain, LabeledOptions);
   
    pData = [pData(IDX.train, :); pData(IDX.test,:)];    
    
    % save data in appropriate format
    Data{idomain} = [];
    
    Data{idomain}.XTrain = XLabeled;
    Data{idomain}.XTest = XTest;
    Data{idomain}.YTrain = YLabeled;
    Data{idomain}.YTest = YTest;
    Data{idomain}.X.labeled = XLabeled;
    Data{idomain}.X.unlabeled = XUnlabeled;
    Data{idomain}.Y.labeled = YLabeled;
    Data{idomain}.Y.unlabeled = YUnlabeled;
    Data{idomain}.spatialData = pData;
    Data{idomain}.trainPrct = options.trainPrct;
    Data{idomain}.labelPrct = options.labelPrct;


end
end