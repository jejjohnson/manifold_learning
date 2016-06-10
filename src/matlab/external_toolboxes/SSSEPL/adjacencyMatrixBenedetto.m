function A = adjacencyMatrixBenedetto(varargin)
% adjacencyMatrixBenedetto - construct adjacency matrix from spectral/spatial 
%   data using Benedetto method
% usage: A = adjacencyMatrixBenedetto(specData,spatData,knnVal,sigmaF,sigmaP,beta);
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
%   sigmaF (scalar) - standard deviation parameter for weighted diffusion
%       metric on spectral data. Default sigmaF = 1.
%   sigmaP (scalar) - standard deviation parameter for weighted diffusion
%       metric on spatial data. Default sigmaP = 1.
%   beta (scalar) - tradeoff parameter between spectral and spatial data.
%       Default beta = 0.5.
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
% date: 1 February 2014

% get/check inputs
[specData,spatData,knnVal,sigmaF,sigmaP,beta,numNodes] = ...
    parseInputs(varargin{:});

% fuse spectral and spatial data
data = [specData.*(sqrt(beta)/sigmaF),spatData.*(sqrt(1-beta)/sigmaP)];

% find k nearest neighbors - remove first column to avoid self-neighbors
[idx,distVal] = knnsearch(data,data,'k',knnVal+1);
idx = idx(:,2:end);
distVal = distVal(:,2:end);

% compute weights
w = exp(-(distVal.^2));

% create sparse weighted adjacency matrix
A = sparse(repmat((1:numNodes)',[1 knnVal]),idx,w,numNodes,numNodes);

% make adjacency matrix symmetric, so that two nodes are connected if
% one of them is within the k nearest neighbors of the other
A = max(A,A.');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subfunction parseInputs
function [specData,spatData,k,sigmaF,sigmaP,beta,N] = parseInputs(varargin)

% get/check number of inputs
nargs = numel(varargin);
narginchk(1,6);

% get/check spectral data
specData = varargin{1};
if ~isequal(ndims(specData),2) || ~isa(specData,'double')
    error([mfilename,':parseInputs:badSpecData'],'Spectral data must be 2-D double precision array');
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

% get/check sigmaF
if nargs>3
    sigmaF = varargin{4};
else
    sigmaF = [];
end
if isempty(sigmaF)
    sigmaF = 1;
end
if ~isscalar(sigmaF) || (sigmaF<=0)
    error([mfilename,':parseInputs:badSigmaF'],'sigmaF must be positive scalar');
end

% get/check sigmaP
if nargs>4
    sigmaP = varargin{5};
else
    sigmaP = [];
end
if isempty(sigmaP)
    sigmaP = 1;
end
if ~isscalar(sigmaP) || (sigmaP<=0)
    error([mfilename,':parseInputs:badSigmaP'],'sigmaP must be positive scalar');
end

% get/check beta
if nargs>5
    beta = varargin{6};
else
    beta = [];
end
if isempty(beta)
    beta = 0.5;
end
if ~isscalar(beta) || (beta<0) || (beta>1)
    error([mfilename,':parseInputs:badBeta'],'beta must be scalar between zero and one inclusive.');
end
