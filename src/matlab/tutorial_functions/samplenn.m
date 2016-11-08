function W = samplenn(data, Options)
%==========================================================================
% NEARESTNEIGHBORS finds the nearest neighbor distances from the
% datapoints. 
%
% Examples
% --------
%
% This first example uses the exhaustive MATLAB knn model and then performs
% a k-nearest neighbor search.
%
% >> data = rand(10,5);
% >> Options.alg = 'exhaustive';
% >> Options.nnGraph = 'knn';
% >> Options.k = 20;
% >> [indices, distances] = samplenearestneighors(data, Options);
%
% This second example uses the KDTree MATLAB knn model and then performs a
% radius-nearest neighbor search.
%
% >> data = rand(10,5);
% >> Options.alg = 'kdtree';
% >> Options.nnGraph = 'r';
% >> Options.r = 5;
% >> [indices, distances] = nearestneighbors(data, Options);
%
% Parameters
% ----------
% * data    - 2D array (N x D)
%               contains the data points where N is the number of data
%               points and D is the number of dimensions.
% * Options - struct (optional)
%               contains the parameters to modify the nearest neighbor
%               search. If no options are included, then the default 
%               parameters are used. The fields are as follows:
%       * nnMethod - The available nearest neighbor methods are:
%           + 'knn' 
%           + 'radius'
%           + 'connected'
%       * nnGraph - The way to do the graph.
%           + 'normal'
%           + 'mutual'
%       * k - number of nearest neighbors for graph (default = 20).
%       * r - radius for range search for nearest neighbors (default = 20).
%           
% Returns
% -------
% * indices - array (N x K)%               contains the indices where all of the k or r nearest points
%               can be found.
% * distances - array (N x K)
%               conatins the distance values of all of the k or r nearest
%               points can be found.
%
% Information
% -----------
% * Author      : J. Emmanuel Johnson
% * Email       : emanjohnson91@gmail.com
% * Date        : 8-Nov-2016
%
% References
% ----------
% * MATLAB Central - Fast Spectral Clustering
%       https://goo.gl/zzfN8T
%==========================================================================

nSamples = size(data, 2);



switch nnMethod
    case 'knn'
    
        % Preallocate memory
        indicesI = zeros(1, kNeighbors * nSamples);
        indicesJ = zeros(1, kNeighbors * nSamples);
        indicesZ = zeros(1, kNeighbors * nSamples);
    
        for ii = 1:n
        
            % Compute i-th column of distance matrix
            dist = distEuclidean(repmat(data(:, ii), 1, nSamples), data);

            % Sort row by distance
            [s, O] = sort(dist, 'ascend');

            % Save indices and value of the k 
            indicesI(1, (ii-1) * kNeighbors+1:ii*kNeighbors) = ii;
            indicesJ(1, (ii-1) * kNeighbors+1:ii*kNeighbors) = O(1:kNeighbors);
            indicesZ(1, (ii-1) * kNeighbors+1:ii*kNeighbors) = s(1:kNeighbors);
            
        end
        
        % Create sparse matrix
        W = sparse(indicesI, indicesJ, indicesZ, nSamples, nSamples);

        % clear large variables
        clear indicesI indicesJ indicesZ dists s O
      
    case 'radius'
    
        for ii = 1:n
            % Compute the i-th column of distance matrix
            dist = distEuclidean(repmat(data(:, ii), 1, nSamples), data);

            % Find distances smaller than epsilon (unweighted)
            dist = (dist < epsilon);

            % Now save the indices and values for the adjacency matrix
            lastind   = size(indicesI,2);
            count     = nnz(dist);
            [~, col]  = find(dist);

            indicesI(1, lastind+1:lastind+count) = ii;
            indicesJ(1, lastind+1:lastind+count) = col;
            indicesZ(1, lastind+1:lastind+count) = 1;
        end      
        % Create sparse matrix
        W = sparse(indicesI, indicesJ, indicesZ, nSamples, nSamples);

        % clear large variables
        clear indicesI indicesJ indicesZ dists s O
    
    case 'connected'
        
        W = squareform(pdist(data'));
    
    otherwise
        error([mfilename, 'samplenn:badnnmethod'], ...
            'Unrecognized nearest neighbor method.');
end





% Construct either normal or mutual graph
switch nnGraph
    case 'normal'
        % Normal
        W = max(W, W');
    case 'mutual'
        % Mutual
        W = min(W, W');
    otherwise
        error([mfilename, 'samplenn:badgraphtype'], ...
            'Unrecognized graph type.');
end


end
