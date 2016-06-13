function [embedding, lambda] = linear_graph_embedding(W, X, options)


%==========================================================================
% Check Inputs
%==========================================================================
[W, X, options] = parseInputs(W, X, options);

%==========================================================================
% Tune the Eigenvalue problem
%==========================================================================

switch options.type
    
    case 'le'
        % compute the Laplacian Matrix
        D = spdiags(sum(W,2), 0, size(W,1), size(W,1));
        A = D-W;
        
        % compute the constraint matrix
        if strcmp(options.constraint, 'degree')
            B = D;
        elseif strcmp(options.constraint, 'identity')
            B = speye(size(W,1));
        else
            error('Unrecognized constraint matrix')
            
        end
        

        
    case 'ssse'
        
        % compute Laplacian matrix
        D = spdiags(sum(W,2), 0, size(W,1), size(W,1));
        L = D-W;
        

        % compute spatial-spectral potential
        V = SpatialSpectralPotential(options.weightdata,...
            options.clusterdata, ...
            options.idx,...
            options.ssse);
        
        % Tune the trade off
        A = L + options.alpha * trace(L)./trace(V) * V;
        
        %------------------------------
        % compute the constraint matrix
        %------------------------------
        
        % diagonal degree matrix
        if strcmp(options.constraint, 'degree')
            B = D;
            
        % identity matrix
        elseif strcmp(options.constraint, 'identity')
            B = speye(size(W,1));
        else
            error('Unrecognized constraint matrix')
            
        end
        
        
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