function V = schroedingerSpatialPotential(varargin)
% schroedingerSpatialPotential - construct potential matrix for encoding
%   spatial information in Schroedinger operator
% usage: V = schroedingerSpatialPotential(img,valFlag);
%    or: V = schroedingerSpatialPotential(img,valFlag,sigma);
%
% arguments:
%   img (numRows x numCols x numSpectra) - image with which to construct
%       potential matrix
%   valFlag (boolean scalar) - specifies whether to create nondiagonal
%       potentials using Gilles-Bowles weight (false) or Shi-Malik weight
%       (true). Default valFlag = true.
%   sigma (scalar) - standard deviation parameter for weighted diffusion
%       metric on spectral data. Default sigma = 1. Only used if valFlag is
%       true.
%
%   V (sparse NxN) - matrix of nondiagonal potentials encoding spatial
%       information for Schroedinger operator
%

% author: Nathan D. Cahill
% email: nathan.cahill@rit.edu
% date: 14 October 2013

% get/check inputs
[img,valFlag,sigma,numRows,numCols] = parseInputs(varargin{:});

% create subscripts for image positions, padded with single row/column
[i,j] = ndgrid(2:numRows+1, 2:numCols+1);
ind = sub2ind([numRows+2, numCols+2],i,j);

% initialize sparse matrix V
szSp = (numRows+2)*(numCols+2);
V = spalloc(szSp, szSp, 6*(numRows+2)*(numCols+2));

% north neighbors
indNew = sub2ind([numRows+2, numCols+2],i-1,j);
vals = zeros(numRows,numCols);
if valFlag
    vals(2:numRows,:) = exp(-sum((img(2:numRows,:,:) - img(1:numRows-1,:,:)).^2,3)./(sigma^2));
else
    vals(2:numRows,:) = exp(-acos( sum(img(2:numRows,:,:).*img(1:numRows-1,:,:),3)./...
        ( sqrt(sum(img(2:numRows,:,:).^2,3)).*sqrt(sum(img(1:numRows-1,:,:).^2,3)) ) ));
end
V = V + sparse(ind,indNew,-vals,szSp,szSp);

% west neighbors
indNew(:) = sub2ind([numRows+2, numCols+2],i,j-1);
vals = zeros(numRows,numCols);
if valFlag
    vals(:,2:numCols) = exp(-sum((img(:,2:numCols,:) - img(:,1:numCols-1,:)).^2,3)./(sigma^2));
else
    vals(:,2:numCols) = exp(-acos( sum(img(:,2:numCols,:).*img(:,1:numCols-1,:),3)./...
        ( sqrt(sum(img(:,2:numCols,:).^2,3)).*sqrt(sum(img(:,1:numCols-1,:).^2,3)) ) ));
end
V = V + sparse(ind,indNew,-vals,szSp,szSp);

% south neighbors
indNew(:) = sub2ind([numRows+2, numCols+2],i+1,j);
vals = zeros(numRows,numCols);
if valFlag
    vals(1:numRows-1,:) = exp(-sum((img(1:numRows-1,:,:) - img(2:numRows,:,:)).^2,3)./(sigma^2));
else
    vals(1:numRows-1,:) = exp(-acos( sum(img(1:numRows-1,:,:).*img(2:numRows,:,:),3)./...
        ( sqrt(sum(img(1:numRows-1,:,:).^2,3)).*sqrt(sum(img(2:numRows,:,:).^2,3)) ) ));
end
V = V + sparse(ind,indNew,-vals,szSp,szSp);

% east neighbors
indNew(:) = sub2ind([numRows+2, numCols+2],i,j+1);
vals = zeros(numRows,numCols);
if valFlag
    vals(:,1:numCols-1) = exp(-sum((img(:,1:numCols-1,:) - img(:,2:numCols,:)).^2,3)./(sigma^2));
else
    vals(:,1:numCols-1) = exp(-acos( sum(img(:,1:numCols-1,:).*img(:,2:numCols,:),3)./...
        ( sqrt(sum(img(:,1:numCols-1,:).^2,3)).*sqrt(sum(img(:,2:numCols,:).^2,3)) ) ));
end
V = V + sparse(ind,indNew,-vals,szSp,szSp);

% remove padded rows/columns from V, scale by 1/e
V = V(ind,ind)/exp(1);

% compute diagonal elements of V
V = spdiags(-sum(V).',0,V);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subfunction parseInputs
function [img,valFlag,sigma,numRows,numCols] = parseInputs(varargin)

% get/check number of inputs
nargs = numel(varargin);
narginchk(1,3);

% get/check image
img = varargin{1};
numRows = size(img,1);
numCols = size(img,2);

% get/check valFlag
if nargs>1
    valFlag = varargin{2};
else
    valFlag = [];
end
if isempty(valFlag)
    valFlag = true;
end
if ~islogical(valFlag) || ~isscalar(valFlag)
    error([mfilename,':parseInputs:badValFlag'],'valFlag must be boolean scalar');
end

% get/check sigma
if nargs>2
    sigma = varargin{3};
else
    sigma = [];
end
if isempty(sigma)
    sigma = 1;
end
if ~isscalar(sigma) || (sigma<=0)
    error([mfilename,':parseInputs:badSigma'],'sigma must be positive scalar');
end
