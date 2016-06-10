function A = adjacencyMatrixShiMalik(varargin)
% adjacencyMatrixShiMalik - construct adjacency matrix from hyperspectral 
%   image using Shi-Malik algorithm
% usage: A = adjacencyMatrixShiMalik(imageData,rad,sigmaF,sigmaP);
%
% arguments:
%   imageData (MxNxD) - image containing hyperspectral data (MxN image, 
%       D components per pixel).  
%   rad (positive scalar) - Radius (in pixels) of circular neighborhood in
%       which to compute weights for adjacency matrix. Default rad = 2.
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
% date: 1 February 2014

% get/check inputs
[imageData,rad,sigmaF,sigmaP,M,N,D] = parseInputs(varargin{:});

% compute pixel offsets for all pixels in a rad-neighborhood of the origin
radInt = floor(rad);
[colOffset,rowOffset] = meshgrid(-radInt:radInt,-radInt:radInt);
neighborhoodMask = ((colOffset.^2 + rowOffset.^2) <= radInt.^2);
neighborhoodMask(radInt+1,radInt+1) = false;
colOffset = colOffset(neighborhoodMask);
rowOffset = rowOffset(neighborhoodMask);
distOffset = sqrt(colOffset.^2 + rowOffset.^2);
numNeighbors = numel(colOffset);

% initialize adjacency matrix
A = spalloc(M*N, (M+2*radInt)*(N+2*radInt), numNeighbors*M*N);

% pad image array
imageDataPad = padarray(imageData,[radInt radInt 0]);

% reshape image and padded image data
imageDataVec = reshape(imageData,M*N,D);
imageDataPadVec = reshape(imageDataPad,(M+2*radInt)*(N+2*radInt),D);

% initialize indices into padded image for each row of adjacency matrix
[colInd,rowInd] = meshgrid((1:N)+radInt,(1:M)+radInt);
colInd = colInd(:);
rowInd = rowInd(:);

% precompute squared values of sigmas
sigmaF2 = sigmaF.^2;
sigmaP2 = sigmaP.^2;

% loop through numNeighbors, constructing adjacency matrix
ind = (1:M*N)';
for j = 1:numNeighbors
    
    % first, compute weight for spatial distance
    spatWeight = exp(-(distOffset(j).^2)/sigmaP2);
    
    % next, compute weights for spectral distance
    newInd = sub2ind([M+2*radInt N+2*radInt], rowInd + rowOffset(j), colInd + colOffset(j));
    specWeight = exp(-sum((imageDataVec - imageDataPadVec(newInd,:)).^2,2)/sigmaF2);
    
    % now, create sparse matrix with Shi-Malik weights for these offsets,
    % and add to A
    A = A + sparse(ind, newInd, ...
        spatWeight.*specWeight, M*N, (M+2*radInt)*(N+2*radInt));
    
end

% delete columns from A corresponding to padded elements
A = A(:,sub2ind([M+2*radInt N+2*radInt],rowInd,colInd));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subfunction parseInputs
function [imageData,rad,sigmaF,sigmaP,M,N,D] = parseInputs(varargin)

% get/check number of inputs
nargs = numel(varargin);
narginchk(1,4);

% get/check image data
imageData = varargin{1};
if (ndims(imageData)<2) || (ndims(imageData)>3) || ~isa(imageData,'double')
    error([mfilename,':parseInputs:badImageData'],'Image data must be 3-D double precision array');
end
[M,N,D] = size(imageData);

% get/check radius
if nargs>1
    rad = varargin{2};
else
    rad = [];
end
if isempty(rad)
    rad = 2;
end
if ~isscalar(rad) || (rad<1)
    error([mfilename,':parseInputs:badRadius'],'Radius must be scalar greater than or equal to one.');
end

% get/check sigmaF
if nargs>2
    sigmaF = varargin{3};
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
if nargs>3
    sigmaP = varargin{4};
else
    sigmaP = [];
end
if isempty(sigmaP)
    sigmaP = 1;
end
if ~isscalar(sigmaP) || (sigmaP<=0)
    error([mfilename,':parseInputs:badSigmaP'],'sigmaP must be positive scalar');
end
