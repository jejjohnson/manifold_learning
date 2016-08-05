function [varargout] = alignment(Data, options)

% get data
nSamplesTotal = []; 
dDomainsTotal = [];
X = cell(numel(Data));
Y = cell(numel(Data));
Domains = cell(numel(data));
Samples = cell(numel(data));
YBlock = [];
Z = [];

for idomain = 1:numel(Data)
    % labeled and unlabeled matrix
    X{idomain} = [Data{idomain}.X.labeled, Data{idomain}.X.unlabeled];
    
    % block data matrix
    Z = sparseblkdiag(Z, X{idomain});
    
    Y{idomain} = [Data{idomain}.Y.Labeled, ...
        zeros(size(Data{idomain}.X.unlabeled), 1)];
    
    YBlock = [YBlock, Y{idomain}];
    

    
        
    [Domains{idomain}, Samples{idomain}] = size(X{idomain});
    
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
    
    A{idomain} = Adjacency(X{idomain}, options);
    
    P = sparseblk([P, Potential(A{idomain}, options)]);  % Block Potential Matrix
    W = sparseblk([W, A{idomain}]);                             % Block Adjacency Matrix
    
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
WDissimilarity = sparse(repmat(YBlock, 1, length(YBlock))==...
    repmat(YBlock, 1, length(YBlock))');
WDissimilarity(YBlock == 0, :) = 0;
WDissimilarity(:, YBlock == 0) = 0;
WDissimilarity = double(WDissimilarity);

% SSMA Voodoo
if isequal(options.type, 'ssma')
    Sws = sum(sum(WSimilarity));
    Sw = sum(sum(W));
    WSimilarity = WSimilarity/Sws * Sw;
    
    Swd = sum(sum(WDissimilarity));
    WDissimilarity = WDissimilarity/Swd * Sw;
end

% Save matrices
if options.printing == 1
    figure;
    spy(W)
    print('..', '-depsc2')
    
    figure;
    spy(WSimilarity)
    print('..', '-depsc2')
    
    figure;
    spy(WDissimilarity)
    print('..', '-depsc2')

end


% Diagonal and Laplacian Matrices
DSimilarity = sum(WSimilarity, 2);
LSimilarity = diag(DSimilarity)-WSimilarity;

D = sum(W,2);
L = diag(D) - W;

DDissimilarity = sum(WDissimilarity, 2);
LDissimilarity = diag(DDissimilarity) - WDissimilarity;

% Alignment Tuner
switch lower(options.type)
    
    case 'ssma'
        
        A = ((1-options.mu) * L ) + options.mu * LSimilarity + ...
            options.lambda * eye(size(L));
        A = Z' * A * Z;
        B = LDissimilarity + options.lambda * eye(size(LDissimilarity));
        B = Z' * B * Z;
        
    case 'wang'
        
         A  = ((1-options.mu) * L ) + options.mu * LSimilarity + ...
            options.lambda * eye(size(L));
        
        A = Z' * A * Z;
        
        B = Z' * D * Z;
        
    case 'sssma'
        
        for idomain = 1:numel(Data)
             
            % Block Potential Matrix
            P = sparseblk([P, Potential(A{idomain}, options)]);  
        end
        
         A  = (L + options.alpha * P) + options.mu * LSimilarity + ...
            options.lambda * eye(size(L));
        
        A = Z' * A * Z;
        
        B = Z' * D * Z;
        
        
    otherwise
        error('Unrecognized Alignment type.');
        
end

%======================
% Generalized Problem
%======================

        

end
