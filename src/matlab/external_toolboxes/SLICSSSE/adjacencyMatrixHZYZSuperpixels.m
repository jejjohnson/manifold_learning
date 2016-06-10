function A = adjacencyMatrixHZYZSuperpixels(varargin)
% adjacencyMatrixHZYZSuperpixels - construct adjacency matrix from 
%   superpixels of an image using HZYZ algorithm
% usage: A = adjacencyMatrixHZYZSuperpixels(spectralData,supPixPosition,k,sigmaF,sigmaP);
%
% arguments:
%   spectralData (MxN) - N-dimensional feature vector corresponding to the mean 
%       spectral values in each of M superpixels extracted from an image
%   supPixPosition (Mx2) - average row/column pixel positions for each of
%       the M superpixels.
%   k (positive integer) - number of nearest neighbors. Default k = 20.
%   sigmaF (postive scalar) - standard deviation parameter for weighted 
%       diffusion metric on spectral data. Default sigmaF = 1.
%   sigmaP (postive scalar) - standard deviation parameter for weighted 
%       diffusion metric on spatial data. Default sigmaP = 1.
%
%   A (sparse MNxMN) - weighted adjacency matrix
%
% Note: for consistency, it is recommended to normalize imageData so that 
%   the norm of the vector at each pixel has an average value of 1. This 
%   function will do NO normalization, however.
%

% author: Nathan D. Cahill
% email: nathan.cahill@rit.edu
% date: 28 July 2014

% get/check inputs
[spectralData,supPixPosition,knnVal,sigmaF,sigmaP,M] = parseInputs(varargin{:});

% fuse spectral and spatial data
data = [spectralData./sigmaF,supPixPosition./sigmaP];

% find k nearest neighbors - remove first column to avoid self-neighbors
idx = knnsearch(data,data,'k',knnVal+1,'Distance',@distHZYZ);
idx = idx(:,2:end);

% create sparse weighted adjacency matrix
A = sparse(repmat((1:M)',[1 knnVal]),idx,ones(M,knnVal),M,M);

% make adjacency matrix symmetric, so that two nodes are connected if
% one of them is within the k nearest neighbors of the other
A = max(A,A.');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subfunction parseInputs
function [spectralData,supPixPosition,k,sigmaF,sigmaP,M,D] = parseInputs(varargin)

% get/check number of inputs
nargs = numel(varargin);
narginchk(2,5);

% get/check spectral data
spectralData = varargin{1};
if (ndims(spectralData)>2) %|| size(LabData,2)~=3
    error([mfilename,':parseInputs:badspectralData'],'Spectral data must be Mx3 double precision array');
end
M = size(spectralData,1);

% get/check position data
supPixPosition = varargin{2};
if (ndims(supPixPosition)>2) || (size(supPixPosition,2)~=2) ||(size(supPixPosition,1)~=M)
    error([mfilename,':parseInputs:badSupPixPosition'],'Superpixel position must be Mx2 double precision array');
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
    error([mfilename,':parseInputs:badK'],'k must be scalar greater than or equal to one.');
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


