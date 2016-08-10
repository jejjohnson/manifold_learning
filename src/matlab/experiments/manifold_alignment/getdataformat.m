function Data = getdataformat(ImageData, options)

% number of domains
nDomains = numel(ImageData);
Data = cell(1,nDomains);            % Initialize Data cell for storage


for idomain = 1:nDomains
    
    
    
    % get labeled and unlabeled data;
    LabeledOptions = [];
    LabeledOptions.trainPrct = options.labelPrct{idomain};
    
    [XLabeled, YLabeled, XTest, YTest, IDX] = ...
        traintestsplit(...
        ImageData{idomain}.imgVec, ...
        ImageData{idomain}.gtVec, ...
        LabeledOptions);
    
    % get spatial locations of matrix
    [x, y] = meshgrid(1:ImageData{idomain}.dims.numCols, ...
        1:ImageData{idomain}.dims.numRows);
    pData = [x(:) y(:)];

    pDataLabel = pData(IDX.train, :);
    
    % get unlabeled training and unlabeled testing
    TrainTestOptions = [];
    TrainTestOptions.trainPrct = options.trainPrct{idomain};
    
    [XUnlabeled, YUnlabeled, ~, ~, IDX] = ...
        traintestsplit(XTest, YTest, TrainTestOptions);
    
    
    pDataUnlabel = pData(IDX.train, :);
    
    pData = [pDataLabel; pDataUnlabel];
    
    
    
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