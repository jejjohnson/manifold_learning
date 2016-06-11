function [projection, lambda] = LocalityPreservingProjections(...
    W, data, options)
%{

LaplacianEigenmaps
    [embedding, lambda] = LaplacianEigenmaps(W, options, data)

        
Parameters
----------
W - (N x N) matrix of samples x samples size
    a weighted affinity/connectivity/adjacency matrix 

options - a matlab structure data structure with fields in options to be
          set. See GraphEmbedding for more options.
        
        n_components - the reduced dimensionality of the data
        
        constraint - the normalization parameter
                    * 'identity'
                    * 'degree'
                    * 'k_scaling' (TODO)
                    * 'dissimilarity' (TODO)

Returns
-------
embedding - (N x d) matrix of sample projection x number of components
            the projected embedding of the eigenvalue decomposition

lambda - (d) column vector as the sorted eigenvalues

References
----------

Deng Cai's subspace


Written by Juan Emmanuel Johnson

%}


%==========================================================================
% Construct the Core Elements
%==========================================================================

% Construct the Diagonal Degree matrix

D = spdiags(sum(W,2), 0, size(W,1), size(W,1));

% Construct the Spectral laplacian Matrix
L = D-W;

%==========================================================================
% Construct the Constraint Matrix for Eigenvalue Decomposition
%==========================================================================
if (~isfield(options, 'constraint'))
    options.constraint = 'degree';
end

switch options.constraint
    case 'degree'
        B = D;
    case 'identity'
        B = speye(size(L,1));
end


%==========================================================================
% Create Feature Matrix
%==========================================================================

A = sparse(data'*L*data);
B = sparse(data'*B*data);

%==========================================================================
% Perform Eigenvalue Decomposition
%==========================================================================
if (~isfield(options, 'n_components'))
    options.n_components = 20;
end

size(A)
% perform eigenvalue decomposition (we want the smallest ones)
[projection, lambda] = eigs(A, B, round(1.5*(options.n_components)),'SM');

% discard return N_components
projection = projection(:, 1:options.n_components);
lambda = diag(lambda); lambda=lambda(1:options.n_components);


end