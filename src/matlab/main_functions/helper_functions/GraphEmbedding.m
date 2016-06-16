function [embedding, lambda] = GraphEmbedding(W, options)
%
% GraphEmbedding
%   Function that takes an adjacency matrix and solves the Eigenvalue
%   decomposition problem with a few extra options to tune the problem.
%
% Usage
% -----
%   [embedding, lambda] = GraphEmbedding(W, options)
%
% Parameters
% ----------
% W     -   array (N x N); typically n_samples by n_samples
%   an adjacency/connectivity/affinity matrix to be used for the
%   eigenvalue decomposition step.
% options   -   a matlab struct with options for tuning the problem
%   
%       * type  - indicates the type of tuning the function allows
%                     current options include:
%                       - 'le' [default]
%                       - 'ss' (spatial-spectral)
%                       - 'pl' (partial-labels)
%                       - 'sspl' (spatial-spectral and partial labels)
%
%       * embedding     - int [default = 20]
%           number of eigenvalues and eigenvectors to retain from
%           the eigenvalue decomposition problem.
%       * ss    - matlab struct with options for ss potential
%           See SpatialSpectralPotential.m for more details
%
%       * pl    - matlab struct with options for pl potential
%           See PartialLabelsPotential.m for more details
% 
% Returns
% -------
% embedding     - array (N x k), typically of reduced dimension k
%       contains the eigenvectors of the generalized eigenvalue
%       decomposition problem.
% lambda        - vector (k x 1), 
%       contains the eigenvalues of the generalized eigenvalue 
%       decomposition problem.
%   
% Information
% -----------
% Author    : Juan Emmanuel Johnson
% Email     : jej2744@rit.edu
% Date      : 11 June 2016
%
% TODO
% ----
% * Documentation   
%   - Examples
% * Function Features
%   - add different normalized laplacian matrices
%   - add different constraint matrices
%   

%==========================================================================
% Check Inputs
%==========================================================================

[W, options] = parseInputs(W, options);

%==========================================================================
% compute Laplacian matrix and potentials
%==========================================================================

% compute standard Laplacian matrix
D = spdiags(sum(W,2), 0, size(W,1), size(W,1));
L = D-W;

switch options.type
    
    % standard laplacian eigenmaps
    case 'le'
        % compute the Laplacian Matrix
        A = L;
        
    % spatial-spectral schroedinger eigenmaps 
    case 'ss'
        
        % compute spatial-spectral potential
        V = SpatialSpectralPotential(options.ss.weightdata,...
            options.ss.clusterdata, ...
            options.ss.idx,...
            options.ss);
        
        % Tune the trade off
        A = L + options.ss.alpha * trace(L)./trace(V) * V;
    
    % partial-labels schroedinger eigenmaps
    case 'pl'

        % compute the partial labels potential
        V = PartialLabelsPotential(options.pl.labels, ...
            options.pl);
        
        % tune the trade off
        A = L + options.pl.beta * trace(L)./trace(V) * V;
        
    % spatial-spectral w/ partial-labels schroedinger eigenmaps
    case 'sspl'
        
        % compute the partial labels potential
        Vss = PartialLabelsPotential(options.pl.labels, ...
            options.pl);
        
        % compute spatial-spectral potential
        Vpl = SpatialSpectralPotential(options.ss.weightdata,...
            options.ss.clusterdata, ...
            options.ss.idx,...
            options.ss);
        
        % tune trade off
        A = L + options.ss.alpha* trace(L)./trace(Vss) * Vss + ...
            options.pl.beta * trace(L)./trace(Vpl) * Vpl;
        
    otherwise 
        error('Unrecognized Laplacian tuner.')
           
end

%==========================================================================
% Compute the constraint matrix
%==========================================================================

% diagonal degree matrix
switch options.constraint
    
    % standard diagonal degree matrix
    case 'degree'
        B = D;
     
    % identity matrix
    case 'identity'
        
        B = speye(size(W,1));
        
    otherwise
        error('Unrecognized constraint matrix')
end

%==========================================================================
% Perform Eigenvalue Decomposition
%==========================================================================


[embedding, lambda] = eigs(A, B, ...
            round(1.5*(options.n_components+1)),'SA');
        
% discard smallest eigenvalue
embedding = embedding(:, 2:options.n_components+1);
lambda = diag(lambda); lambda=lambda(2:options.n_components+1);        


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Subfunction - parseInputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [W, options] = parseInputs(W, options)
    
%--------------------------------------------------------------------------
% mandatory options
%--------------------------------------------------------------------------

% check entry matrix for dimensions and precision
if ~isequal(ndims(W),2) || ~isa(W, 'double') 
    error([mfilename, ':parseInputs:baddata'], ...
        'data must be 2-D double precision array');
end

% check entry matrix for dimensions
if ~isequal(size(W,1), size(W,2))
    error([mfilename, ':parseInputs:baddata'], ...
        'data dimensions must be equal');
end

%--------------------------------------------------------------------------
% default options
%--------------------------------------------------------------------------

% if no potential type [default - 'le']
if ~isfield(options, 'type')
    options.type = 'le';            % standard laplacian Eigenmaps
end

% constraint matrix
if ~isfield(options, 'constraint')
    options.constraint = 'degree';
end

% number of components
if ~isfield(options, 'n_components')
    options.n_components = round(size(W,1)/4);
end

% eigenvalue decomposition method
if ~isfield(options, 'decomposition_method')
    options.decomposition_method = 'matlab';
end

%--------------------------------------------------------------------------
% check inputs for special cases
%--------------------------------------------------------------------------

switch lower(options.type)
    
    case 'le'
        
    
    case 'ss'
        
        %----------------------------
        % check required options
        %----------------------------
        
        % required partial labels data
        if ~isfield(options, 'ss') 
            error([mfilename, ':parseInputs:nossdata'],...
                'for ss case, must have ss data.');
        end
        
        % required spectral data spatial points
        if ~isfield(options.ss, 'clusterdata') || ...
                ~isfield(options.ss, 'weightdata')
            error([mfilename, ':parseInputs:noinput'], ...
                'Schroedinger Eigenmaps requires two data inputs.');
        end
        
        % trade off parameter for ss potential
        if ~isfield(options.ss, 'alpha')
            options.ss.alpha = 17.78;
        elseif ~isfloat(options.ss.alpha) || options.ss.alpha < 0
            error([mfilename, ':parseInputs:badalphadata'],...
                'Alpha value must be a float greater than 0.');
        end
        
        % kernel for spatial data
        if ~isfield(options.ss, 'clusterkernel')
            options.ss.clusterkernel = 'heat';     % default heat kernel
        end
        
        % sigma value for spatial
        if ~isfield(options.ss, 'clusterSigma')
            options.ss.clusterSigma = 1.0;
        end
        
        % sigma value for spectral
        if ~isfield(options.ss, 'weightSigma')
            options.ss.weightSigma = 1.0;
        end
        
        % required indices of knn
        if ~isfield(options.ss, 'idx')
            error([mfilename, ':parseInputs:noinput'], ...
                'Schroedinger Eigenmaps requires a nn index');
        end
        
    case 'pl'
        
        %----------------------------
        % check required options
        %----------------------------
        
        % required partial labels data
        if ~isfield(options, 'pl') 
            error([mfilename, ':parseInputs:noydata'],...
                'for pl case, must have pl data.');
            
        % check partial labels data types and size
        elseif ~isvector(options.pl.labels) 
            error([mfilename, ':parseInputs:badlabelsdata'], ...
                'y must be a vector with double inputs.');
        end
        
        % make sure we have W if constraint Conditioning is true
        if isfield(options.pl, 'constraintConditioning') 
            disp('options...constraint')
            options.pl.W = W;
        end
        
        % check active constraints
        if ~isfield(options.pl, 'mlactive')
            options.pl.mlactive = ...
                1:numel(unique(options.pl.labels(options.pl.labels>0)));
        elseif ~isvector(options.pl.mlactive) || ...
                numel(options.pl.mlactive) > ...
                numel(unique(options.pl.labels(options.pl.labels>0))) || ...
                nume(options.mlactive) < 0
            error([mfilename, ':parseInputs:badmlactivedata'],...
                'mlactive must be a vector of size 0 to # of classes.');
        end
        
        % trade off parameter for ss potential
        if ~isfield(options.pl, 'beta')
            options.pl.beta = 17.78;
        elseif ~isfloat(options.pl.beta) || options.pl.beta < 0
            error([mfilename, ':parseInputs:badbetadata'],...
                'beta value must be a float greater than 0.');
        end
        
    case 'sspl'
        
        %----------------------------
        % check required options
        %----------------------------
        
        % required partial labels data
        if ~isfield(options, 'ss') 
            error([mfilename, ':parseInputs:nossdata'],...
                'for ss case, must have ss data.');
        end
        
        % required spectral data spatial points
        if ~isfield(options.ss, 'clusterdata') || ...
                ~isfield(options.ss, 'weightdata')
            error([mfilename, ':parseInputs:noinput'], ...
                'Schroedinger Eigenmaps requires two data inputs.');
        end
        
        % trade off parameter for ss potential
        if ~isfield(options.ss, 'alpha')
            options.ss.alpha = 17.78;
        elseif ~isfloat(options.ss.alpha) || options.ss.alpha < 0
            error([mfilename, ':parseInputs:badalphadata'],...
                'Alpha value must be a float greater than 0.');
        end
        
        % kernel for spatial data
        if ~isfield(options.ss, 'clusterkernel')
            options.ss.clusterkernel = 'heat';     % default heat kernel
        end
        
        % sigma value for spatial
        if ~isfield(options.ss, 'clusterSigma')
            options.ss.clusterSigma = 1.0;
        end
        
        % sigma value for spectral
        if ~isfield(options.ss, 'weightSigma')
            options.ss.weightSigma = 1.0;
        end
        
        % required indices of knn
        if ~isfield(options.ss, 'idx')
            error([mfilename, ':parseInputs:noinput'], ...
                'Schroedinger Eigenmaps requires a nn index');
        end
        
        %----------------------------
        % check required options for partial labels
        %----------------------------
        
        % required partial labels data
        if ~isfield(options, 'pl') 
            error([mfilename, ':parseInputs:noydata'],...
                'for pl case, must have pl data.');
            
        % check partial labels data types and size
        elseif ~isvector(options.pl.labels) 
            error([mfilename, ':parseInputs:badlabelsdata'], ...
                'y must be a vector with double inputs.');
        end
        
        % make sure we have W if constraint Conditioning is true
        if isfield(options.pl, 'constraintConditioning') 
            disp('options...constraint')
            options.pl.W = W;
        end
        
        % check active constraints
        if ~isfield(options.pl, 'mlactive')
            options.pl.mlactive = ...
                1:numel(unique(options.pl.labels(options.pl.labels>0)));
        elseif ~isvector(options.pl.mlactive) || ...
                numel(options.pl.mlactive) > ...
                numel(unique(options.pl.labels(options.pl.labels>0))) || ...
                nume(options.mlactive) < 0
            error([mfilename, ':parseInputs:badmlactivedata'],...
                'mlactive must be a vector of size 0 to # of classes.');
        end
        
        % trade off parameter for ss potential
        if ~isfield(options.pl, 'beta')
            options.pl.beta = 17.78;
        elseif ~isfloat(options.pl.beta) || options.pl.beta < 0
            error([mfilename, ':parseInputs:badbetadata'],...
                'beta value must be a float greater than 0.');
        end
        
    otherwise
        error([mfilename, ':parseInputs:badtypedata'],...
                'Must choose le, ss, pl or sspl.');
        
        

end
        

    
end
end