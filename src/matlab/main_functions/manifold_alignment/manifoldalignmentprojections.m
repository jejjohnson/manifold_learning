function [varargout] = manifoldalignmentprojections(Data, projections, type)

% Get Data from parameters
nSamplesTotal = 0; 
dDomainsTotal = 0;
X = cell(numel(Data));
Y = cell(numel(Data));
Domains = cell(numel(Data));
Samples = cell(numel(Data));
YBlock = [];
Z = [];

for idomain = 1:numel(Data)
    % labeled and unlabeled matrix
    X{idomain} = [Data{idomain}.X.labeled; Data{idomain}.X.unlabeled];
    
    % block data matrix
    Z = sparse(blkdiag(Z, X{idomain}));
    
    Y{idomain} = [Data{idomain}.Y.labeled; ...
        zeros(size(Data{idomain}.X.unlabeled, 1), 1)];
    
    YBlock = [YBlock; Y{idomain}];
        
    [Samples{idomain}, Domains{idomain}] = size(X{idomain});
    
    dDomainsTotal = Domains{idomain} + dDomainsTotal;           % total number of dimensions
    nSamplesTotal = Samples{idomain} + nSamplesTotal;            % total number of samples
end


% Project Data
domainProjections = cell(1,numel(Data));
embedding = cell(1, numel(Data));


startPoint = 1; endPoint = Domains{1};
for idomain = 1:numel(Data)
    
    % grab appropriate size dimensions
    domainProjections{idomain} = projections(:, startPoint:endPoint);
    
    % project training and testing data
    embedding{idomain}.train = Data{idomain}.X.labeled * domainProjections{idomain}';
    embedding{idomain}.test = Data{idomain}.XTest * domainProjections{idomain}';
    
    % normalize the data if ssma method
    switch type
        case 'ssma'
            % normalize the data if ssma
            meanProj = mean(embedding{idomain}.train);
            stdProj = std(embedding{idomain}.train);
    
            embedding{idomain}.train = zscore(embedding{idomain}.train);
    
            T = length(Data{idomain}.XTest)/2;
    
            embedding{idomain}.test = ((embedding{idomain}.test - ...
                repmat(meanProj, 2*T,1)) ./ repmat(stdProj, 2*T, 1));
    end
    
    % Try for the next iteration
    try
        startPoint = Domains{idomain}+1;
        endPoint = Domains{idomain} + Domains{idomain+1};
    % if fails, break the loop
    catch
        break
    end
end


%=================%
% Outputs
%=================%

switch nargout
    case 1
        varargout{1} = embedding;
    case 2
        varargout{1} = embedding;
        varargout{2} = domainProjections;
    otherwise
        error('Incorrect number of outputs.');
end

end