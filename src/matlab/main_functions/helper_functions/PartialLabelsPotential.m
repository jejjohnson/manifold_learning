function V = PartialLabelsPotential(y, options)
% PartialLabelsPotential 
%   Constructs a potential matrix with labels.
%
% Usage
% -----
%   V = PartialLabelsPotential(y,options)
%
% Parameters
% ----------
% y     -   vector ( N x 1)
%   A vector of labeled values. Assumption that zero is an unlabeled
%   samples.
%
% options   -   a matlab struct of optional values
% 
%   *   constraintConditioning  :   boolean [true or false]
%           declares whether constraint conditioning is used or not.
%           
%   *   W   -   array (N x N)
%           contains the adjacency matrix from the sample matrix X.
%           
%   *   mlactive    -   vector ( 1 x n )
%           contains the class labels that the user would like to
%           include in the partial labels potential matrix.
%           (Note: will give an error if n > N where N is the number of
%           classes found in the y vector)
%
% Returns
% -------
% V     :   array (N x N)
%   A matrix containing the partial labels laplacian matrix.
%
% Information
% -----------
% Version   : v1
% Author    : Nathan D. Cahill
% Email     : nathan.cahill@rit.edu
% Date      : 5-Jul-14
% Version   : v2
% Author    : Juan Emmanuel Johnson
% Email     : jej2744@rit.edu
% Date      : 15-Jun-16
% 
% TODO
% ----
% * Documentation 
%       - References
%       - Examples
% * Function Features
%       - different graph constructions
%

%==========================================================================
% Check Inputs
%==========================================================================

[y, options] = parseInputs(y, options);

%==========================================================================
% check labels
%==========================================================================

% find the number of classes greater than 0
numClasses = numel(unique(y(y>0)));

%==========================================================================
% Construct struct with class labels
%==========================================================================

% construct a cell to hold all of the partial labels
MLcell = cell(numClasses,1);

% loop through all of the class labels
for i = 1:numClasses
    
    % find the indices that have label i
    ind = find(y(y==i));
    
    % construct a complete graph w/ those indices
    MLcell{i} = completeGraph(ind);
    
end

%==========================================================================
% Choose appropriate class labels
%==========================================================================

% concatenate the ML constraints for active classes
ML = cat(1, MLcell{options.mlactive});

% create U matrix
U = sparse(repmat((1:size(ML,1))', [1 2]), ML, ...
    [ones(size(ML,1), 1), -ones(size(ML, 1), 1)], ...
    size(ML, 1), size(y,1));

%==========================================================================
% Create potential matrix
%==========================================================================


% include constrain (optional)
if isfield(options, 'constraintConditioning')
    Dinv = spdiags(1./sum(options.W,2), 0, ...
        size(options.W,1), size(options.W,1));
    U = U * Dinv * options.W;
end

% create partial label cluster potential
V = U' * U;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Subfunction - parseInputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [y, options] = parseInputs(y, options)
   
%--------------------------------------------------------------------------
% check mandatory data
%--------------------------------------------------------------------------
if ~isequal(size(y,2),1) || ~isfloat(y)
    error([mfilename, ':parseInputs:badydata'], ...
        'y must be a vector with double inputs.');
end

%--------------------------------------------------------------------------
% check options
%--------------------------------------------------------------------------

% make sure we have W if constraint Conditioning is true
if isfield(options, 'constraintConditioning') && ~isfield(options, 'W')
    error([mfilename, ':parseInputs:noWdata'],...
        'To do constraint Conditioning, need adjacency matrix W.')
end

% check W size matches size y
if isfield(options, 'constraintConditioning')
    if ~isequal(size(options.W,1), size(y,1))
        error([mfilename, ':parseInputs:badWdata'],...
            'size(W,1) must be equal to size(y,1).');
    end
    

    
end

% if there are no active constraint options
if ~isfield(options, 'mlactive')
    options.mlactive = 1:numel(unique(y(y>0)));
elseif ~isvector(options.mlactive) || ...
        numel(options.mlactive) > numel(unique(y(y>0))) || ...
        numel(options.mlactive) < 0
    error([mfilename, ':parseInputs:badmlactivedata'],...
        'mlactive must be a vector of size 0 to # of classes.');
end

    
    
    
end

end