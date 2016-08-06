function Data = getdataformat(ImageData, options)

% number of domains
nDomains = numel(ImageData);
Data = cell(1,nDomains);            % Initialize Data cell for storage


for idomain = 1:nDomains
    
    TrainTestOptions = [];
    TrainTestOptions.trainPrct = options.trainPrct;
    
    % training and testing split
    [XTrain, YTrain, XTest, YTest, IDX] = ...
        traintestsplit(...
        ImageData{idomain}.imgVec, ...
        ImageData{idomain}.gtVec, ...
        TrainTestOptions);
    
    % get spatial locations of matrix
    [x, y] = meshgrid(1:ImageData{idomain}.dims.numCols, ...
        1:ImageData{idomain}.dims.numRows);
    pData = [x(:) y(:)];

    pData = pData(IDX.train, :);
    
    % get labeled and unlabeled data;
    LabeledOptions = [];
    LabeledOptions.trainPrct = options.labelPrct;
    
    [XLabeled, YLabeled, XUnlabeled, YUnlabeled] = ...
        traintestsplit(XTrain, YTrain, LabeledOptions);
    
    % save data in appropriate format
    Data{idomain} = [];
    
    Data{idomain}.XTrain = XTrain;
    Data{idomain}.XTest = XTest;
    Data{idomain}.YTrain = YTrain;
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