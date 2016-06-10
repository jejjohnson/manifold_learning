function [A,idx] = adjacencyMatrix(varargin)
% adjacencyMatrix - construct adjacency matrix from spectral/spatial data
% usage: A = adjacencyMatrix(specData,spatData,knnVal,sigma,eta,gamma,angleFlag);
%
% arguments:
%   specData (NxD) - array of spectral data (N data points, D spectral
%       values).  If specData = [], only spatial data will be used to
%       construct adjacency matrix.
%   spatialData (NxM) - array of spatial data (N data points, M 
%       dimensions). If spatData = [], only spectral data will be used to
%       construct adjacency matrix.
%   knnVal (scalar) - number of nearest neighbors to choose in k-nearest
%       neighbors algorithm for finding graph edges. Default knnVal = 20.
%   sigma (scalar) - standard deviation parameter for weighted diffusion
%       metric on spectral data. Default sigma = 1.
%   eta (scalar) - standard deviation parameter for weighted diffusion
%       metric on spatial data. Default eta = 1.
%   gamma (scalar) - weighting factor between spectral and spatial data for
%       fusion metric. Default gamma = D/M.
%   angleFlag (logical scalar) - true specifies using exp(angle) as the
%       weight. false specifies heat kernel.  Default angleFlag = false.
%
%   A (sparse NxN) - weighted adjacency matrix
%
% Note: for consistency, it is recommended to normalize specData so that 
%   all spectral data is in the range [0,1], and to normalize spatial data
%   so that each pixel is 1 unit from its neighbor. This function will do
%   NO normalization, however.
%

% author: Nathan D. Cahill
% email: nathan.cahill@rit.edu
% date: 14 October 2013

% get/check inputs
[specFlag,specData,spatFlag,spatData,knnVal,sigma,eta,gamma,angleFlag,numNodes] = ...
    parseInputs(varargin{:});

% fuse spectral and spatial data if necessary
if specFlag
    if spatFlag
        data = [specData,gamma.*spatData];
    else
        data = specData;
    end
else
    if spatFlag
        data = spatData;
    else
        A = [];
        return
    end
end

% find k nearest neighbors - remove first column to avoid self-neighbors
if angleFlag
    distType = 'cosine';
else
    distType = 'euclidean';
end
[idx,distVal]=knnsearch(data,data,'k',knnVal+1,'Distance',distType);
idx = idx(:,2:end);
distVal = distVal(:,2:end);

% compute weights
if angleFlag
    w = exp(-acos(1-distVal));
else
    if specFlag
        w = exp(-(distVal.^2)./(sigma.^2));
    else
        w = exp(-(distVal.^2)./(eta.^2));
    end
end

% create sparse weighted adjacency matrix
A = sparse(repmat((1:numNodes)',[1 knnVal]),idx,w,numNodes,numNodes);

% make adjacency matrix symmetric, so that two nodes are connected if
% one of them is within the k nearest neighbors of the other
A = max(A,A.');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subfunction parseInputs
function [specFlag,specData,spatFlag,spatData,k,sigma,eta,gamma,angleFlag,N] = parseInputs(varargin)

% get/check number of inputs
nargs = numel(varargin);
narginchk(1,7);

% get/check spectral data
specData = varargin{1};
if ~isequal(ndims(specData),2) || ~isa(specData,'double')
    error([mfilename,':parseInputs:badSpecData'],'Spectral data must be 2-D double precision array');
end
if isempty(specData)
    specFlag = false;
else
    specFlag = true;
end
N = size(specData,1);

% get/check spatial data
if nargs>1
    spatData = varargin{2};
else
    spatData = [];
end
if ~isequal(ndims(spatData),2) || ~isa(spatData,'double')
    error([mfilename,':parseInputs:badSpatData'],'Spatial data must be 2-D double precision array');
end
if isempty(spatData)
    spatFlag = false;
else
    spatFlag = true;
end
if (N==0)
    N = size(spatData,1);
else
    if ~ismember(size(spatData,1),[0 N]);
        error([mfilename,':parseInputs:badSpatDataSize'],'Spatial data must be empty or same size as spectral data');
    end
end

% get/check k
if nargs>2
    k = varargin{3};
else
    k = [];
end
if isempty(k)
    k = 20;
end
if ~isscalar(k) || (k<1) || ~isequal(round(k),k)
    error([mfilename,':parseInputs:badK'],'k must be scalar integer');
end

% get/check sigma
if nargs>3
    sigma = varargin{4};
else
    sigma = [];
end
if isempty(sigma)
    sigma = 1;
end
if ~isscalar(sigma) || (sigma<=0)
    error([mfilename,':parseInputs:badSigma'],'sigma must be positive scalar');
end

% get/check eta
if nargs>4
    eta = varargin{5};
else
    eta = [];
end
if isempty(eta)
    eta = 1;
end
if ~isscalar(eta) || (eta<=0)
    error([mfilename,':parseInputs:badEta'],'eta must be positive scalar');
end

% get/check gamma
if nargs>5
    gamma = varargin{6};
else
    gamma = [];
end
if isempty(gamma)
    gamma = max(size(specData,2),1)/max(size(spatData,2),1);
end
if ~isscalar(gamma) || (gamma<=0)
    error([mfilename,':parseInputs:badGamma'],'gamma must be positive scalar');
end

% check special case where both spectral and spatial data are supplied - 
%   sigma and eta must be equal
if (specFlag&&spatFlag)
    if ~isequal(sigma,eta)
        warning('If both spectral and spatial data are supplied, sigma and eta must be equal. We will continue by using eta = sigma.');
    end
end

% get/check angleFlag
if nargs>6
    angleFlag = varargin{7};
else
    angleFlag = [];
end
if isempty(angleFlag)
    angleFlag = false;
end
if ~isscalar(angleFlag) || ~islogical(angleFlag)
    error([mfilename,':parseInputs:badAngleFlag'],'angleFlag must be logical scalar');
end
