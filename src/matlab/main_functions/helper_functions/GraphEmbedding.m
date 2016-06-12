function [embedding, lambda] = GraphEmbedding(W, options)


%==========================================================================
% Check Inputs
%==========================================================================
[W, options] = parseInputs(W, options);

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
        elseif strcomp(options.constraint, 'identity')
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
        elseif strcomp(options.constraint, 'identity')
            B = speye(size(W,1));
        else
            error('Unrecognized constraint matrix')
            
        end
        
        
end

%==========================================================================
% Perform Eigenvalue Decomposition
%==========================================================================


[embedding, lambda] = eigs(A, B, ....
            round(1.5*(options.n_components+1)),'SA');
        
% discard smallest eigenvalue
embedding = embedding(:, 2:options.n_components+1);
lambda = diag(lambda); lambda=lambda(2:options.n_components+1);        


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Subfunction - parseInputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [W, options] = parseInputs(W, options)
    
% if no type
if ~isfield(options, 'type')
    options.type = 'le';            % standard laplacian Eigenmaps
end

% check entry matrix for dimensions and precision
if ~isequal(ndims(W),2) || ~isa(W, 'double') 
    error([mfilename, ':parseInputs:baddata'], ...
        'data must be 2-D double precision array');
end

% check entry matrix for 
if ~isequal(size(W,1), size(W,2))
    error([mfilename, ':parseInputs:baddata'], ...
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
    options.n_components = round(W/4);
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