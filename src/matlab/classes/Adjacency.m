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
% ~ Example which uses the default settings and just creates an adjacency
%   matrix
% >> X = data;
% >> A = Adjacency(X); 
%
% - Example which uses the struct file
% >> Options = struct;
% >> Options.alg = 'kdtree';
% >> Options.nnGraph = 'radiusNN';
% >> Options.eRadius = 2;
% >> Options.kernel = 'heat';
% >> A = Adjacency(A, Options);
%
% Properties
% ----------
% * alg         - algorithm used for the nearest neighbor calculations
%                 ('kdtree', 'exhaustive')
% * nnGraph     - type of nearest neighbor graph (kNN, radiusNN, complete).
% * k  - number of nearest neighbors for the k-Nearest Neighbors
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
% * defaultsettings     - static method that produces a default Settings
%                         struct for the Adjacency class.
%
% Information
% -----------
% * Author  : J. Emmanuel Johnson
% * Email   : emanjohnson91@gmail.com
% * Date    : Sat. 5th Nov, 2016
%
%==========================================================================    
properties (Access = public)

    kernelType;
    nnGraph;
    k;
    distance;
    eRadius;
    kernel;
    sigma;
    nnTime;
    X;
    W;
    alg;

end

properties (SetAccess = private, GetAccess = public)

    savedData;
    nnOptions;
    Settings; 

end

methods (Access = public)

    % CONSTRUCTOR
    function self = Adjacency(data, varargin)
        
        % check for input parameters
        narginchk(1,2);
        
        % Parse inputs for settings
        self.parseinputs(data, varargin);
    end
    
    % GET ADJAENCY MATRIX
    function W = getadjacency(self)
        
        % Find the Nearest Neighbors
        [idx, distVals] = self.nearestneighbors(self.X, self.Settings);
        
        % Kernelize Distances
        w = self.distancekernel(distVals, self.Settings);
        
        % Construct Adjacency Matrix
        W = self.constructadjacency(self.X, idx, w);
        
        self.W = W;
        
    end
    
    % DISPLAY ADJACENCY MATRIX
    function displayadjacency(self)
        
        % If adjacency matrix has not been calculated
        if isempty(self.W)
            % Calculate Adjacency Matrix
            self.W = getadjacency(self);
        end
        
        figure;
        spy(self.W);
        
    end

end

methods (Access = private)

    % Parse Inputs
    function parseinputs(self, data, InputArgs)
        %==================================================================
        %
        % Parse the Settings struct for the following inputs:
        % * nnGraph     - type of nearest neighbor graph (kNN, radiusNN, complete).
        % * k  - number of nearest neighbors for the k-Nearest Neighbors
        %                 method.
        % * eRadius     - size of the radius for the radius-nearest neighbors 
        %                 graph.
        % * distance    - distance measurement for the kNN and eNN graph.
        % * kernel      - distance kernel for nearest neighbor method.
        % * sigma       - parameter for heat kernel.
        % * nnTime      - time taken to find the nearest neighbors.
        % * saveNN      - option to use the saved nearest neighbor results.
        % * refNN       - flag to let the class know to save the function.
        % * matType     - determines the type of matrix
        %
        %==================================================================
        
        % Check for existence of Options
        if isempty(InputArgs)               % Check for empty 
            % Create empty Options struct
            Options = struct;
        elseif ~isstruct(InputArgs{1})      % Check if type struct
            error([mfilename, 'parseInputs:badSettingsfile'], ...
                'Incorrect Settings file for Adjacency class.');
        else
            Options = InputArgs{1};
        end
        
        % Check if data is 2D, numeric
        classes = {'numeric'};
        attributes = {'2d'};
        validateattributes(data, classes, attributes);
        
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
        addOptional(p, 'eRadius', defaultr, @isnumeric);      
        
        % distance measurement for kNN and eNN graph
        expecteddistance = {'euclidean'};
        defaultdistance = expecteddistance{1};
        addParameter(p, 'distance', defaultdistance, ...
            @(x) any(validatestring(x, expecteddistance)));
        
        % Kernel for NN distances
        expectedkernel = {'heat', 'cosine'};
        defaultkernel = expectedkernel{1};
        addParameter(p, 'kernel', defaultkernel, ...
            @(x) any(validatestring(x, expectedkernel)));
        
        % Sigma Value for Heat kernel
        defaultSigma = 1.0;
        errorMsg = 'Sigma must be positive and numeric.';
        validateSigma = @(x) assert(isnumeric(x) && (x>0), errorMsg);
        addParameter(p, 'sigma', defaultSigma, ...
            validateSigma);
        
% p = inputParser;
% paramName = 'myparam';
% default = 1;
% errorMsg = 'Value must be positive, scalar, and numeric.'; 
% validationFcn = @(x) assert(isnumeric(x) && isscalar(x) ...
%     && (x > 0),errorMsg);
% addParameter(p,paramName,default,validationFcn)
        
%         % Save NN Results Flag
%         if ~isfield(Options.saveNN)
%             self.saveNN = 0;
%         elseif isequal(Options.saveNN, 1, Options.saveNN, 0)
%             self.saveNN = Options.saveNN;
%         else
%             error([mfilename, 'parseinputs:badSaveNN'], ...
%                 'Incorrect saveNN field input.');
%         end
%         
%         % reference NN matlab results struct
%         if ~isfield(Options.refNN)
%             self.refNN = 0;
%         elseif exist(Options.refNN, 'file')
%             self.refNN = Options.refNN;
%         else
%             error([mfilename, 'parseinputs:badRefNN'], ...
%                 'Incorrect or unrecognized refNN field input.');
%         end
        
        
        % Parse struct
        parse(p, Options);
        
        % Set Properties to Class Instance Accordingly
        self.Settings = p.Results;
        self.alg = p.Results.alg;
        self.nnGraph = p.Results.nnGraph;
        self.k = p.Results.k;
        self.eRadius = p.Results.eRadius;
        self.distance = p.Results.distance;
        self.kernel = p.Results.kernel;
        self.sigma = p.Results.sigma;
        self.nnTime = 0;
        
        self.X = data;
        
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
    narginchk(1,2);

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
            [indices, distances] = knnsearch(KModel, data, 'k', Options.k+1);

        case 'radius'

            % RANGE SEARCH
            [indices, distances] = rangesearch(KModel, data, Options.r);

        otherwise 
            error([mfilename, 'nearestneighbors:unrecognizednnmethod'], ...
                'Unrecognized NN method. Must use "knn" or "radius"');
    end

    % DISPLACE OLD POINTS
    indices = indices(:, 2:end);
    distances = distances(:, 2:end);

    % VARIABLE OUPUTS
    switch nargout
        case 1
            varargout{1} = indices;
        case 2
            varargout{1} = indices;
            varargout{2} = distances;
        otherwise
            error([mfilename, 'nearestneighbors:badnumoutputs'], ...
                'Incorrect number of outputs.');
    end
    
    %------------------------------------------------%
    % SubFunction - Nearest Neighbor Function Parser %
    %------------------------------------------------%
    function Options = parseinputs(Inputs)
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
        if isempty(Inputs{1})
            % Create empty Options struct
            Options = struct;
        elseif ~isstruct(Inputs{1})
            error([mfilename, 'parseInputs:badOptionsfile'], ...
                'Incorrect Options input for nearestneighbors function.');
        else
            Options = Inputs{1};
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
        defaultNNGraph = expectedNNGraph{1};            % default - radius
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

    % parse inputs
    Options = parseinputs(varargin);
    
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
    switch nargout
        case 1
            varargout{1} = scaledDistances;
    end

    %--------------------------------------%
    % SubFunction - Kernel Function Parser %
    %--------------------------------------%
    function Options = parseinputs(Inputs)
        %==================================================================
        %
        % PARSEINPUTS checks for the following parameters:
        % * kernel
        % * sigma
        %
        %==================================================================
        
        % Check for existence of Options
        if isempty(Inputs{1})
            % Create empty Options struct
            Options = struct;
        elseif ~isstruct(Inputs{1})
            error([mfilename, 'parseInputs:badOptionsfile'], ...
                'Incorrect variable input for kernel function.');
        else
            Options = Inputs{1};
        end
        
        % Intiate inputParse class
        p = inputParser;
        
        % Kernel Function
        expectedkernels = {'heat', 'cosine'};
        defaultAlgorithm = expectedkernels{1};       % default - kdtree
        addParameter(p, 'kernel', defaultAlgorithm, ...
            @(x) any(validatestring(x, expectedkernels)));
        
        % Sigma for Heat Kernel
        paramName = 'sigma';
        defaultSigma = 1.0;         % default - 1.0
        errorMsg = 'Value must be positive and numeric.';
        validationFcn = @(x) ...
            assert(isnumeric(x) && (x > 0), errorMsg);
        addOptional(p, paramName, defaultSigma, validationFcn); 
        
        
        % Parse struct
        parse(p, Options);
        
        % Graph Results as Options struct
        Options = p.Results;
        
    end

    end
    
    % DEFAULT SETTINGS FILE
    function Settings = defaultsettings()
    %======================================================================
    %
    % DEFAULTSETTINGS gives some default parameters when there is no
    % settings file during the class construction. 
    %
    % Parameters
    % ----------
    % * alg         - nearest neighbor algorithm
    % * nnGraph     - type of nearest neighbor graph 
    % * k  - number of nearest neighbors for the k-Nearest 
    %                 Neighbors method.
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
    %======================================================================
    
    % Initialize setting struct
    Settings = struct;
    
    % Assign defaults for Settings fields
    Settings.alg = 'kdtree';            % Nearest-Neighbor algorithm
    Settings.nnGraph = 'radiusNN';      % Nearest-Neighbor Graph
    Settings.k = 10;           % k-Nearest Neighbors
    Settings.eRadius = 10;              % e-Radius Neighbors
    Settings.distance = 'euclidean';    % Distance metric for neighbors
    Settings.kernel = 'heat';           % Kernel Function
    Settings.sigma = 1.0;               % Sigma Parameter for k-NN
    Settings.nnTime = 0;                % Time for NN to compute
    Settings.saveNN = 0;                % Save flag for NN
    Settings.refNN = 'na';              % Save location for NN mat file
    Settings.matType = 'sparse';        % Matrix type

    end
    
    % CONSTRUCT ADJACENCY MATRIX
    function W = constructadjacency(data, idx, w)
        
        % construct a sparse adjacency matrix
        W = sparse(repmat((1:size(data, 1))', [1 size(idx,2)]), ...
            idx, w, size(data, 1), size(data, 1));

        % make the matrix symmetric
        W = max(W, W');
        
    end

end
    
end