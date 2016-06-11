function V = schroedingerPotential(varargin)
% schroedingerPotential - construct potential matrix for encoding
%   cluster information in Schroedinger operator
% usage: V = schroedingerPotential(EData,CData,valFlag,sigma,IDX);
%
% arguments:
%   EData (N x M) - data used to define edge weights (N data points, M
%       dimensions)
%   CData (N x P) - data used to define cluster potential weights (N data 
%       points, P dimensions)
%   valFlag (boolean scalar) - specifies whether to create nondiagonal
%       potentials using Gilles-Bowles weight (false) or Shi-Malik weight
%       (true). Default valFlag = true.
%   sigma (1x2) - standard deviation parameters for EData and CData. 
%       Default sigma = [1 1]. sigma(1) is only used if valFlag is
%       true.
%   IDX (N x K) - IDX(i,:) are the indices into the rows of EData and CData
%       that correspond to the K nearest neighbors of the ith data point
%
%   V (sparse NxN) - matrix of nondiagonal potentials encoding spatial
%       information for Schroedinger operator
%

% author: Nathan D. Cahill
% email: nathan.cahill@rit.edu
% date: 12 August 2014

% get/check inputs
[EData,CData,valFlag,sigma,IDX] = parseInputs(varargin{:});

% number of data points
N = size(EData,1);

% number of cluster potentials for each data point
K = size(IDX,2);

% compute weights for EData
x1 = repmat(permute(EData,[1 3 2]),[1 K]);
x2 = reshape(EData(IDX(:),:),[N K size(EData,2)]);
if valFlag % heat kernel
    WE = exp(-sum((x1-x2).^2,3)./(sigma(1).^2));
else % angle
    WE = exp(-acos(sum(x1.*x2,3)./(sqrt(sum(x1.^2,3)).*sqrt(sum(x1.^2,3)))));
end

% compute weights for CData
diffVals = repmat(permute(CData,[1 3 2]),[1 K]) - ...
    reshape(CData(IDX(:),:),[N K size(CData,2)]);
WC = exp(-sum(diffVals.^2,3)./(sigma(2).^2));

% construct nondiagonal elements of V
V = sparse(repmat((1:N)',[1 K]),IDX,-WE.*WC,N,N);

% make symmetric
V = V + V.';

% compute diagonal elements of V
V = spdiags(-sum(V).',0,V);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subfunction parseInputs
function [EData,CData,valFlag,sigma,IDX] = parseInputs(varargin)

% get/check number of inputs
narginchk(5,5);

% get/check EData
EData = varargin{1};
N = size(EData,1);
if ~ismatrix(EData)
    error([mfilename,':BadEData'],'EData must be N x M');
end

% get/check CData
CData = varargin{2};
if ~ismatrix(CData) || (size(CData,1)~=N)
    error([mfilename,':BadCData'],'CData must be N x P, where N = size(EData,1).');
end

% get/check valFlag
valFlag = varargin{3};
if isempty(valFlag)
    valFlag = true;
end
if ~islogical(valFlag) || ~isscalar(valFlag)
    error([mfilename,':parseInputs:badValFlag'],'valFlag must be boolean scalar');
end

% get/check sigma
sigma = varargin{4};
if isempty(sigma)
    sigma = [1 1];
end
if numel(sigma)~=2 || any(sigma<=0)
    error([mfilename,':parseInputs:badSigma'],'sigma must be 1x2 array of positive scalars');
end

% get/check idx
IDX = varargin{5};
if ~ismatrix(IDX) || (size(IDX,1)~=N)
   error([mfilename,':BadIDX'],'IDX must be N x K, where N = size(EData,1).');
end 