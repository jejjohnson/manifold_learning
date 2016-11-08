function [varargout] = nearestneighbors(data, varargin)
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
% >> [indices, distances] = nearestneighbors(data, Options);
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
%       * alg - the matlab algorithm from the toolboxes that can perform
%       the nearest neighbor search. The available methods are:
%           + kdtree 
%           + exhaustive
%           + flann (TODO)
%       * nnGraph - the nearest neighbor graph. Note: some of the options
%       will depend upon the MATLAB alg chosen. So if there is a conflict
%       of choices, then the function will default to the MATLAB alg field.
%           + 'knn'
%           + 'radius'
%       * k - number of nearest neighbors for graph (default = 20).
%       * r - radius for range search for nearest neighbors (default = 20).
%           
% Returns
% -------
% * indices - array (N x K)
%               contains the indices where all of the k or r nearest points
%               can be found.
% * distances - array (N x K)
%               conatins the distance values of all of the k or r nearest
%               points can be found.
%
% Information
% -----------
% * Author      : J. Emmanuel Johnson
% * Email       : emanjohnson91@gmail.com
% * Date        : 7-Nov-2016
%
%==========================================================================

% check number of input arguments
narginchk(0,1);

% check number of output arguments
nargoutchk(1,2);

%==========================================================================
% NN MODEL TYPE
switch searchType
    
    case 'kdtree'
        % KDTREE SEARCH MODEL
        KModel = KDTreeSearcher(data);
        
    case 'exhaustive'
        
        % initialize the exhaustive search model
        KModel = ExhaustiveSearcher(data);
        
    otherwise
        error([mfilename, 'nearestneighbors:badsearchtype'], ...
            'Unrecognized searchType parameter.');
end


% QUERY TYPE
switch queryType
    
    case 'knn'
    
        % KNN SEARCH
        [indices, distances] = knnsearch(KModel, data, 'k', nn+1);
        
    case 'range'

        % RANGE SEARCH
        [indices, distances] = rangesearch(KModel, data, radius);
        
    otherwise 
        error([mfilename, 'nearestneighbors:unrecognizednnmethod'], ...
            'Unrecognized NN method. Must use "knn" or "range"');
end

% DISPLACE OLD POINTS
indices = indices(:, 2:end);
distances = distances(:, 2:end);

% VARIABLE OUPUTS
switch numel(nargout)
    case 1
        varargout{1} = indices;
    case 2
        varargout{1} = indices;
        varargout{2} = distances;
    otherwise
        error([mfilename, 'nearestneighbors:badnumoutputs'], ...
            'Incorrect number of outputs.');
end
        
end