function [varargout] = manifoldalignment(Data, Options)

% Extract Options
AlignmentOptions = Options.AlignmentOptions;
PotentialOptions = Options.PotentialOptions;
AdjacencyOptions = Options.AdjacencyOptions;


% get data
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

    
%================%
% Build A Matrices %
%================%

W = [];
P = [];
A = cell(numel(Data));

for idomain = 1:numel(Data)
    
    A{idomain} = Adjacency(X{idomain}, AdjacencyOptions);
    
    W = sparse(blkdiag(W, A{idomain}));                             % Block Adjacency Matrix
    
end

% double precision W
W = double(W);

%=================%
% Build Class Graphs %
%=================%

% Similarity 
WSimilarity = sparse(repmat(YBlock, 1, length(YBlock))==...
    repmat(YBlock, 1, length(YBlock))');
WSimilarity(YBlock == 0, :) = 0;
WSimilarity(:, YBlock == 0) = 0;
WSimilarity = double(WSimilarity);


% Dissimilarity 
WDissimilarity = sparse(repmat(YBlock, 1, length(YBlock))~=...
    repmat(YBlock, 1, length(YBlock))');
WDissimilarity(YBlock == 0, :) = 0;
WDissimilarity(:, YBlock == 0) = 0;
WDissimilarity = double(WDissimilarity);

switch AlignmentOptions.type
    case 'ssma'
    WDissimilarity = WDissimilarity + eye(size(WDissimilarity,1));
    WSimilarity = WSimilarity + eye(size(WSimilarity,1));

    % SSMA Voodoo
    Sws = sum(sum(WSimilarity));
    Sw = sum(sum(W));
    WSimilarity = WSimilarity/Sws * Sw;

    Swd = sum(sum(WDissimilarity));
    WDissimilarity = WDissimilarity/Swd * Sw;
end

% Save matrices
if AlignmentOptions.printing == 1
    figure;
    spy(W)
%     print('..', '-depsc2')
    
    figure;
    spy(WSimilarity)
%     print('..', '-depsc2')
    
    figure;
    spy(WDissimilarity)
%     print('..', '-depsc2')

    figure;
    spy(WDissimilarity - WSimilarity);
    
    


end


% Diagonal and Laplacian Matrices
DSimilarity = sum(WSimilarity, 2);
LSimilarity = diag(DSimilarity)-WSimilarity;

D = sum(W,2);
L = diag(D) - W;


if AlignmentOptions.printing == 1
    figure;
    spy(sparse(diag(D)))
%     print('..', '-depsc2')
    
    figure;
    spy(L)
%     print('..', '-depsc2')
    
    
    return

end

DDissimilarity = sum(WDissimilarity, 2);
LDissimilarity = diag(DDissimilarity) - WDissimilarity;

% Alignment Tuner
switch lower(AlignmentOptions.type)
    
    case 'ssma'
        
        A = ((1-AlignmentOptions.mu) * L ) + AlignmentOptions.mu * LSimilarity + ...
            AlignmentOptions.lambda * eye(size(L));
        A = Z' * A * Z;
        B = LDissimilarity + AlignmentOptions.lambda * eye(size(LDissimilarity));
        B = Z' * B * Z;
        
    case 'wang'
        
         A  = ((1-AlignmentOptions.mu) * L ) + AlignmentOptions.mu * LSimilarity + ...
            AlignmentOptions.lambda * eye(size(L));
        
        A = Z' * A * Z;
        
        B = Z' * sparse(diag(D)) * Z;
        
    case 'sema'
        
        P = [];
        for idomain = 1:numel(Data)
             
            % construct the spatial adjacency matrix
            [~, idxP] = Adjacency(Data{idomain}.spatialData, ...
                PotentialOptions.SpatialAdjacency);
            
            % construct potential matrix
            V = SpatialSpectralPotential(...
                X{idomain}, ...
                Data{idomain}.spatialData, ...
                idxP, ...
                PotentialOptions);
            
                
            % creat block diagonal with each domain
            P = sparse(blkdiag(P, V));  

        end
        
         A  = (1-AlignmentOptions.mu) * (L + AlignmentOptions.alpha * P) + ...
             AlignmentOptions.mu * LSimilarity + ...
             AlignmentOptions.lambda * eye(size(L));
        
        A = Z' * A * Z;
        
        B = Z' * sparse(diag(D)) * Z;
        
        
    otherwise
        error('Unrecognized Alignment type.');
        
end

%======================
% Generalized Problem
%======================
if ~isfield(AlignmentOptions, 'nComponents') || ...
        isequal(AlignmentOptions.nComponents, 'default')
    nComponents = dDomainsTotal;
else 
    nComponents = AlignmentOptions.nComponents;
end

warning('off')
[embedding, lambda] = eigs(A, B, nComponents,'SM');
warning('on')
        
% Outputs
switch nargout
    case 1
        varargout{1} = embedding;
    case 2
        varargout{1} = embedding;
        varargout{2} = lambda;
    otherwise
        error('Incorrect number of outputs.');
end

        

end
