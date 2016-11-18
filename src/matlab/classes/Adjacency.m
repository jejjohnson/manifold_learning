classdef Adjacency < handle
%==========================================================================
%
% ADJACENCY is class which allows the user to produce an adjacency matrix 
% from data in the form of (N x D) where N is the number of samples and D
% is the dimension of said samples. Some additional functionality includes
% access to static methods like the nearest neighbor algorithm and the
% distance kernel. This class also allows you to display the final
% adjacency matrix.
% 
% Examples
% --------
%
% Properties
% ----------
% * nnGraph     - type of nearest neighbor graph (kNN, radiusNN, complete).
% * kNeighbors  - number of nearest neighbors for the k-Nearest Neighbors
%                 method.
% * eRadius     - size of the radius for the radius-nearest neighbors 
%                 graph.
% * distance    - distance measurement for the kNN and eNN graph.
% * kernel      - distance kernel for nearest neighbor method.
% * sigma       - parameter for heat kernel.
% * nnTime      - time taken to find the nearest neighbors.
% * saveNN      - option to use the saved nearest neighbor results.
% * refNN       - flag to let the class know to save the function.
% * matType     - determines the type of matrix (
%
% Methods
% -------
% * Adjacency           - constructor for class. this initiates the class 
%                         by passing the data and a struct with options 
%                         into the constructor.
% * nearestneighbors    - static method that provides the nearest neighbors
%                         methods for the data provided
% * distancekernel      - static method to allow you scale the distances
%                         between the nearest neighbors.
% * getAdjacency        - goes through the entire algorithm to get the
%                         final Adjacency matrix.
% * displayMat          - displays the final adjacency matrix.
%
% Information
% -----------
% * Author  : J. Emmanuel Johnson
% * Email   : emanjohnson91@gmail.com
% * Date    : Sat. 5th Nov, 2016
%
%==========================================================================    
properties (Access = public)

    kernelType = 'standard';
    nnGraph = 'knn';
    kNeighbors = 20;
    distance = 'euclidean';
    eRadius = 10;
    kernel = 'heat';
    sigma = 1.0;
    nnTime = 0;

end

properties (SetAccess = public, GetAccess = private)

    savedData = None;
    nnOptions;
    

end

methods (Access = public)

    % CONSTRUCTOR
    function self = Adjacency(Data, Settings)



    end
    
    % GET ADJAENCY MATRIX
    function W = getadjacency(self)
        
    end
    
    % DISPLAY ADJACENCY MATRIX
    function displayadjacency(self)
        
    end

    % DESTRUCTOR


end

methods (Access = private)

    % Parse Inputs
    function self = parseinputs(self, data, Settings)


    end

    % Nearest Neighbor Search
    function [idx, dist] = nnSearch(self, data)

        % query the 


    end

    % KD Tree Search
    function [idx, dist] = kdSearch(self, data)

        tic;

        % initialize the KD Tree Model
        KDModel = KDTreeSearcher(data, ...
            'Distance', self.distance);

        % query the vector for k nearest neighbors
        [idx, dist] = knnsearch(KDModel, data, ...
            'k', self.kNeighbors+1);

        % discard the first distance
        idx = idx(:, 2:end);
        dist = dist(:, 2:end);
        self.nnTime = toc;

    end



    % Construct Adjacency Matrix
    function W = constructadjacency(self, data, idx, w)

        switch self.type

            case 'standard'

                % construct a sparse adjacency matrix
                W = sparse(repmat((1:size(data, 1))', [1 self.x]), ...
                    idx, w, size(data, 1), size(data, 1));

                % make the matrix symmetric
                W = max(W, W');

            case 'similarity'

                Ws_left = sparse(repmat(data, 1, length(data)));

                Ws_right = sparse(Ws_left');

                W = Ws_left == Ws_right;

                % set all the zero values to be zero
                W(data == 0, :) = 0;
                W(:, data == 0) = 0;

                % double precious
                W = double(W);

            case 'dissimilarity'

                Ws_left = sparse(repmat(data,1,length(data)));

                Ws_right = sparse(Ws_left');

                W = Ws_left ~= Ws_right;

                figure;
                spy(W)
                % set all the zero values to be zero
                W(data == 0,:) = 0; 
                W(:,data == 0) = 0; 

                % double precision
                W = double(W);

            otherwise
                error([mfilename, 'constructadjacency:badtype'], ...
                    'Unrecognized adjacency matrix.');
        end


    end

end

methods (Static)

    % NEAREST NEIGHBORS
    function [varargout] = nearestneighbors(data, varargin)
    %======================================================================
    % NEARESTNEIGHBORS finds the nearest neighbor distances from the
    % datapoints. This is a static method that can be called without an
    % instances of the ADJACENCY class.
    %
    % Examples
    % --------
    %
    % This first example uses the exhaustive MATLAB knn model and then 
    % performs a k-nearest neighbor search.
    %
    % >> data = rand(10,5);
    % >> Options.alg = 'exhaustive';
    % >> Options.nnGraph = 'knn';
    % >> Options.k = 20;
    % >> [indices, distances] = ...
    %               Adjacency.nearestneighbors(data, Options);
    %
    % This second example uses the KDTree MATLAB knn model and then 
    % performs a radius-nearest neighbor search.
    %
    % >> data = rand(10,5);
    % >> Options.alg = 'kdtree';
    % >> Options.nnGraph = 'r';
    % >> Options.r = 5;
    % >> [indices, distances] = ....
    %               Adjacency.nearestneighbors(data, Options);
    %
    % Parameters
    % ----------
    % * data    - 2D array (N x D)
    %               contains the data points where N is the number of 
    %               data points and D is the number of dimensions.
    % * Options - struct (optional)
    %               contains the parameters to modify the nearest 
    %               neighbor search. If no options are included, then 
    %               the default parameters are used. The fields are as 
    %               follows:
    %       * alg - the matlab algorithm from the toolboxes that can 
    %               perform the nearest neighbor search. The available 
    %               methods are:
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
    %=================================================================%

    % check number of input arguments
    narginchk(0,1);

    % check number of output arguments
    nargoutchk(1,2);
    
    % parse inputs
    Options = parseinputs(varargin);

    %======================================================================
    % NN MODEL TYPE
    switch Options.alg

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
    switch Options.nnGraph

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
    
    %--------------------------------------%
    % SubFunction - Kernel Function Parser %
    %--------------------------------------%
    function Options = parseinputs(varargin)
        %==================================================================
        %
        % PARSEINPUTS checks for the following parameters:
        % * alg
        % * nnGraph
        % * k
        % * r
        %
        %==================================================================
        
        % Check for existence of Options
        if isempty(varargin)
            % Create empty Options struct
            Options = struct;
        elseif ~isstruct(varargin{1})
            error([mfilename, 'parseInputs:badOptionsfile'], ...
                'Incorrect variable input for nearestneighbors function.');
        else
            Options = varargin{1};
        end
        
        % Intiate inputParse class
        p = inputParser;
        
        % Algorithm Specifics
        expectedAlgorithms = {'kdtree', 'exhaustive'};
        defaultAlgorithm = expectedAlgorithms{1};       % default - kdtree
        addParameter(p, 'alg', defaultAlgorithm, ...
            @(x) any(validatestring(x, expectedAlgorithms)));
        
        % Nearest Neighbor Graph
        expectedNNGraph = {'knn', 'radius'};
        defaultNNGraph = expectedNNGraph{2};            % default - radius
        addParameter(p, 'nnGraph', defaultNNGraph, ...
            @(x) any(validatestring(x, expectedNNGraph)));
        
        % k Parameter for kNN Graph
        defaultk = 10;                                  % default k = 10
        addOptional(p, 'k', defaultk, @isnumeric);      
        
        % r Parameter for eNN Graph
        defaultr = 10;                                  % default r = 10
        addOptional(p, 'r', defaultr, @isnumeric);      
        
        % Parse struct
        parse(p, Options);
        
        % Graph Results as Options struct
        Options = p.Results;
        
    end

    end

    % KERNEL FUNCTIONS
    function [varargout] = distancekernel(distances, varargin)
    %======================================================================
    % 
    % DISTANCEKERNEL takes distance values and uses a kernel to scale
    % said values. 
    % 
    % Examples
    % --------
    %
    % >> distVals= rand(10,5);
    % >> Options = {};
    % >> Options.kernel = 'heat';
    % >> Options.sigma = 1.0;
    % >> [newdistVals] = Adjanecy.distancekernel(distVals, Options);
    %
    % Parameters
    % ----------
    % * distances   - double (N x k)
    %                 number of points by the number of k nearest 
    %                 neighbors found.
    % * Options     - struct 
    %                 has options for the kernels needed. See below:
    %       + kernel    - string, ('heat', 'cosine')
    %                     the kernel function used on the dataset
    %       + sigma     - double
    %                     parameter needed for heat kernel.
    %
    % Information
    % -----------
    % Author    : J. Emmanuel Johnson
    % Email     : emanjohnson91@gmail.com
    % Date      : 17th November, 2016
    %
    %======================================================================

    % Quick I/O check
    narginchk(1,2);
    nargoutchk(1,1);

    % Scale the matrices
    switch Options.kernel

        case 'heat'
            scaledDistances = ...
                exp(-(distances.^2) ./ (Options.sigma.^2));

        case 'cosine'
            scaledDistances = ...
                exp(-acos(1-distances));

        otherwise
            error([mfilename, 'distancekernel:badkernel'], ...
               'Unrecognized distance kernel function.');
    end

    % Outputs
    switch length(narargout)
        case 1
            varargout{1} = scaledDistances;
    end

    %--------------------------------------%
    % SubFunction - Kernel Function Parser %
    %--------------------------------------%
    function Options = parseInputs(varargin)

    end

    end

end
    
end