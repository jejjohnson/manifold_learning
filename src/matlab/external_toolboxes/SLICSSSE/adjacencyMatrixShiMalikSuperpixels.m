function A = adjacencyMatrixShiMalikSuperpixels(varargin)
% adjacencyMatrixShiMalikSuperpixels - construct adjacency matrix from 
%   superpixels of an image using Shi-Malik algorithm
% usage: A = adjacencyMatrixShiMalikSuperpixels(LabData,BHistData,supPixPosition,rad,sigmaF,sigmaP);
%
% arguments:
%   LabData (Mx3) - 3-dimensional feature vector corresponding to the mean 
%       L*a*b* values in each of M superpixels extracted from an image
%   BHistData (MxD) - D-dimensional feature vector corresponding to the 
%       histogram of brightness (L*) in each of M superpixels extracted 
%       from the image
%   supPixPosition (Mx2) - average row/column pixel positions for each of
%       the M superpixels.
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
% date: 28 July 2014

% get/check inputs
[LabData,BHistData,supPixPosition,rad,sigmaF,sigmaP,M,D] = parseInputs(varargin{:});

% compute distance matrix between all pairs of superpixels
DMat = squareform(pdist(supPixPosition));

% create mask to indicate positions where DMat <= rad
DMask = (DMat<=rad) & (DMat>0);

% compute spatial weights
spatWeight = exp((-DMat(DMask).^2)./(sigmaP.^2));

% compute mean Lab value weights
meanLabDistMat = squareform(pdist(LabData));
meanLabWeight = exp((-meanLabDistMat(DMask).^2)./(sigmaF.^2));

% compute chi-square distance between all pairs of brightness histograms
%C2Mat = squareform(pdist(BHistData,@chi2dist));
%BHistWeight = 1 - C2Mat(DMask);
%BHistWeight = exp((-C2Mat(DMask).^2)./(sigmaF.^2));

% construct adjacency matrix
A = spalloc(M,M,sum(DMask(:)));
A(DMask) = spatWeight.*meanLabWeight;
%A(DMask) = spatWeight.*BHistWeight;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subfunction parseInputs
function [LabData,BHistData,supPixPosition,rad,sigmaF,sigmaP,M,D] = parseInputs(varargin)

% get/check number of inputs
nargs = numel(varargin);
narginchk(3,6);

% get/check Lab data
LabData = varargin{1};
if (ndims(LabData)>2) %|| size(LabData,2)~=3
    error([mfilename,':parseInputs:badLabData'],'Lab data must be Mx3 double precision array');
end
M = size(LabData,1);

% get/check brightness histogram data
BHistData = varargin{2};
if (ndims(BHistData)>2) || any(BHistData(:)<0)
    error([mfilename,':parseInputs:badBHistData'],'Brightness histogram data must be MxD double precision array of positive values');
end
[~,D] = size(BHistData);

% get/check position data
supPixPosition = varargin{3};
if (ndims(supPixPosition)>2) || (size(supPixPosition,2)~=2) ||(size(supPixPosition,1)~=M)
    error([mfilename,':parseInputs:badSupPixPosition'],'Superpixel position must be Mx2 double precision array');
end

% get/check radius
if nargs>3
    rad = varargin{4};
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
if nargs>4
    sigmaF = varargin{5};
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
if nargs>5
    sigmaP = varargin{6};
else
    sigmaP = [];
end
if isempty(sigmaP)
    sigmaP = 1;
end
if ~isscalar(sigmaP) || (sigmaP<=0)
    error([mfilename,':parseInputs:badSigmaP'],'sigmaP must be positive scalar');
end
