function [embedding, lambda] = LinearGraphEmbedding(W, X, options)


%==========================================================================
% Check Inputs
%==========================================================================
[W, X, options] = parseInputs(W, X, options);

%==========================================================================
% Tune the Eigenvalue problem
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


[embedding, lambda] = eigs(X'*A*X, X'*B*X, ...
            options.n_components+1,'SM');
               


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Subfunction - parseInputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [W, X, options] = parseInputs(W, X, options)
    
% if no type
if ~isfield(options, 'type')
    options.type = 'le';            % standard laplacian Eigenmaps
end

% check adjacency matrix for dimensions and precision
if ~isequal(ndims(W),2) || ~isa(W, 'double') 
    error([mfilename, ':parseInputs:baddata'], ...
        'data must be 2-D double precision array');
end

% check data matrix for dimensions and precision
if ~isequal(ndims(X),2) || ~isa(X, 'double') 
    error([mfilename, ':parseInputs:badXdata'], ...
        'data must be 2-D double precision array');
end

% check adjacency matrix for equal dimensions
if ~isequal(size(W,1), size(W,2))
    error([mfilename, ':parseInputs:badWdata'], ...
        'data dimensions must be equal');
end

% check adjacency and data matrix for equal dimensions
if ~isequal(size(W,1), size(X,1))
    error([mfilename, ':parseInputs:badXdata'], ...
        'data dimensions must be equal');
end

%--------------------------------------------------------------------------
% default options
%--------------------------------------------------------------------------

% constraint matrix
if ~isfield(options, 'constraint')
    options.constraint = 'degree';
end

% number of components
if ~isfield(options, 'n_components')
    
    options.n_components = round(size(X,2)/4);
elseif options.n_components > size(X,2)
    
    error([mfilename, ':parseInputs:badn_componentsdata'], ...
        'n_components must be <= size(X,2).')
end

% eigenvalue decomposition method
if ~isfield(options, 'decomposition_method')
    options.decomposition_method = 'matlab';
end

%--------------------------------------------------------------------------
% check inputs for special cases
%--------------------------------------------------------------------------

switch lower(options.type)
    
    case 'ssse'
        
        % required spectral data spatial points
        if ~isfield(options, 'clusterdata') || ...
                ~isfield(options, 'weightdata')
            error([mfilename, ':parseInputs:noinput'], ...
                'Schroedinger Eigenmaps requires two data inputs.');
        end
        
        % trade off parameter
        if ~isfield(options, 'alpha')
            options.alpha = 17.78;
        end
        
        % kernel for spatial data
        if ~isfield(options, 'clusterkernel')
            options.ssse.clusterkernel = 'heat';     % default heat kernel
        end
        
        % sigma value for spatial
        if ~isfield(options, 'clusterSigma')
            options.ssse.clusterSigma = 1.0;
        end
        
        % sigma value for spectral
        if ~isfield(options, 'weightSigma')
            options.ssse.weightSigma = 1.0;
        end
        
        % required indices of knn
        if ~isfield(options, 'idx')
            error([mfilename, ':parseInputs:noinput'], ...
                'Schroedinger Eigenmaps requires a nn index');
        end
        

end
        

    
end
end