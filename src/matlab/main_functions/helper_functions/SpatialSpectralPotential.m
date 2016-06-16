function V = SpatialSpectralPotential(WData, CData, idx, options)
% SpatialSpectralPotential - construct potential matrix for encoding
%   cluster information in Schroedinger operator
% usage: V = SpatialSpectralPotential(EData,CData,idx,options);
%
% Parameters
% ----------
% EData (N x M) - data used to define edge weights (N data points, M
%       dimensions)
% CData (N x P) - data used to define cluster potential weights (N data 
%       points, P dimensions)
%   valFlag (boolean scalar) - specifies whether to create nondiagonal
%       potentials using Gilles-Bowles weight (false) or Shi-Malik weight
%       (true). Default valFlag = true.
%   sigma1 float - standard deviation parameter for heat kernel EData
%           [default = 1.0]
%   sigma2 float - standard deviation parameter for heat kernel CData. 
%           [default = 1.0]
%   idx (N x K) - idx(i,:) are the indices into the rows of EData and CData
%       that correspond to the K nearest neighbors of the ith data point
%
%   V (sparse NxN) - matrix of nondiagonal potentials encoding spatial
%       information for Schroedinger operator
%
% Returns
% -------
% 
%
% v1  
% Author: Nathan D. Cahill
% Email: nathan.cahill@rit.edu
% Date: 12 August 2014
% v2
% Author: Juan Emmanuel Johnson
% Email: jej2744@rit.edu
% Date: 11 June 2016

%==========================================================================
% Check Inputs
%==========================================================================

[WData, CData, idx, options] = parseInputs(WData, CData, idx, options);

%==========================================================================
% Find the Potential Matrix
%==========================================================================
% number of data points
N = size(WData,1);

% number of cluster potentials for each data point
K = size(idx,2);

% compute weights for EData
x1 = repmat(permute(WData,[1 3 2]),[1 K]);
x2 = reshape(WData(idx(:),:),[N K size(WData,2)]);

% roll through trade-off cases
switch options.clusterkernel
    case 'heat'
        WE = exp(-sum((x1-x2).^2,3)./(options.weightSigma.^2));
    case 'angle'
        WE = exp(-acos(...
            sum(x1.*x2,3)./(sqrt(sum(x1.^2,3)).*sqrt(sum(x1.^2,3)))));
end

% compute weights for CData
diffVals = repmat(permute(CData,[1 3 2]),[1 K]) - ...
    reshape(CData(idx(:),:),[N K size(CData,2)]);
WC = exp(-sum(diffVals.^2,3)./(options.clusterSigma.^2));

% construct nondiagonal elements of V
V = sparse(repmat((1:N)',[1 K]),idx,-WE.*WC,N,N);

% make symmetric
V = V + V.';

% compute diagonal elements of V
V = spdiags(-sum(V).',0,V);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Subfunction - parseInputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [WData,CData,idx, options] = parseInputs(...
    WData, CData, idx, options)

% check WData 
if ~isequal(ndims(WData),2)
    error([mfilename, ':parseInputs:baddata'], ...
        'WData must be N x M');
end

% check that CData and WData are equal
if ~isequal(ndims(CData),2) || (size(CData,1)~= size(WData,1))
    error([mfilename, ':parseInputs:baddata'], ...
        'CData must be N x P, where N = size(EData,1).');
end

% check the indices size
if ~isequal(ndims(idx),2) || (size(idx,1) ~= size(CData,1))
    error([mfilename, ':parseInputs:baddata'], ...
        'idxp must be N x K, where N = size(CData,1).');
end

   
%--------------------------------------------------------------------------
% default options
%--------------------------------------------------------------------------

% default sigma for heat kernel 1
if ~isfield(options, 'clusterSigma')
    options.clusterSigma = 1.0;
elseif options.clusterSigma <=0
    error([mfilename, ':parseInputs:badclusterSigma1'],...
        'Sigma must be a positive scalar');
end

% default sigma for heat kernel 2
if ~isfield(options, 'weightSigma')
    options.weightSigma = 1.0;
elseif options.weightSigma <=0
    error([mfilename, ':parseInputs:badweightSigma2'],...
        'weightSigma must be a positive scalar');
end
    
    
% default kernel for trade off
if ~isfield(options, 'clusterkernel')
    options.clusterkernel = 'heat';
elseif ~strcmp(options.clusterkernel,'heat') || ...
        ~strcmp(options.clusterkernel,'heat')
    error([mfilename,':parseInputs:badclusterkernel'],...
        'clusterkernel must be either "heat" or "angle".');
end

end
end

