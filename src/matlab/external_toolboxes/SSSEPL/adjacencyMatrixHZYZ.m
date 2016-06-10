function A = adjacencyMatrixHZYZ(varargin)
% adjacencyMatrixHZYZ - construct adjacency matrix from spectral/spatial 
%   data using Hou-Zhang-Ye-Zheng method
% usage: A = adjacencyMatrix(specData,spatData,knnVal,sigmaF,sigmaP);
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
%       metric on spectral data. Default sigma = 1.
%   sigmaP (scalar) - standard deviation parameter for weighted diffusion
%       metric on spatial data. Default eta = 1.
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
[specData,spatData,knnVal,sigmaF,sigmaP,numNodes] = ...
    parseInputs(varargin{:});

% fuse spectral and spatial data
data = [specData./sigmaF,spatData./sigmaP];

% find k nearest neighbors - remove first column to avoid self-neighbors
idx = knnsearch(data,data,'k',knnVal+1,'Distance',@distHZYZ);
idx = idx(:,2:end);

% create sparse weighted adjacency matrix
A = sparse(repmat((1:numNodes)',[1 knnVal]),idx,ones(numNodes,knnVal),numNodes,numNodes);

% make adjacency matrix symmetric, so that two nodes are connected if
% one of them is within the k nearest neighbors of the other
A = max(A,A.');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subfunction parseInputs
function [specData,spatData,k,sigmaF,sigmaP,N] = parseInputs(varargin)

% get/check number of inputs
nargs = numel(varargin);
narginchk(1,5);

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subfunction distHZYZ - Assumes spatial data has two components
function d = distHZYZ(ZI, ZJ) 

d = (1 - exp(-sum((repmat(ZI(1:end-2),[size(ZJ,1),1]) - ZJ(:,1:end-2)).^2,2))).*...
    (1 - exp(-sum((repmat(ZI(end-1:end),[size(ZJ,1),1]) - ZJ(:,end-1:end)).^2,2)));


