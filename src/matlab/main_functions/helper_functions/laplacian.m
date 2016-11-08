function varargout = laplacian(W, varargin)
%{
======================================================================
LAPLACIAN computs the laplacian matrix of an adjacency matrix, W.

Examples
--------

>> Options.lapType = 'unnormalized';
>> L = laplacian(L, Options);

>> [L, D] = laplacian(L, Options);

Parameters
----------
* W   - array (N x N)
          an adjacency matrix representing graph G. Could be sparse
          or dense.
* Options - a MATLAB struct (optional)
      + lapType - str, 
            ~ 'unnormalized' (default)
            ~ 'normalized'
            ~ 'geometric' (TODO)
            ~ 'randomwalk' (TODO)
            ~ 'symmetric' (TODO)
            ~ 'renormalized' (TODO)
      + matType - str, ['dense', 'sparse']
Returns
-------
* Laplacian - array (N x N)
                an array representing the Laplacian matrix
* Degree  - array (N x N)
            an array representing the diagonal degree matrix 

Information
-----------
Author    : J. Emmanuel Johnson
Email     : emanjohnson91@gmail.com
Date      : 11/8/16
======================================================================
%}

%=====================================================================
%-- Degree Matrix --%

% Calculate the degree matrix
degW = sum(W, 2);
D = sparse(1:size(W,1), 1:size(W,2), degW);

% Avoid dividing by zero
degW(degW == 0) = eps;

%=====================================================================
%-- Laplacian Matrix --%

% Compute the Laplacian Matrix
switch laptype
    
    case 'unnormalized'
          
          % Calculate the Normalized Laplacian Matrix
          L = D - W;
    
    case 'normalized'         % shi, malik
          
          % Calculate the inverse of D
          D = spdiags(1 ./ degW, 0 size(D, 1), size(D, 2));
          
          % Calculate the normalized Laplacian Matrix
          L = D * L
          
          % NEED TO DO THIS FOR JORDAN WEISS
          
          % Compute Eigenvectors corresponding to the k smallest eigenvalues
          %diff = eps;
          %[U, ~] = eigs(L, k, diff);
          %
          %% Normalized the eigenvectors row-wise
          %U = bsxfun(@divide, U, sqrt(U .^2, 2));
          
    case 'normalized'          % jordan, weiss
    
          % Calculate D^(-1/2)
          D = spdiags(1./(degW .^ 0.5), 0, size(D,1), size(D,2));
          
          % Calculate the normalized Laplacian
          L = D * L * D;
    otherwise
          error([mfilename, 'laplacian:badlapchoice'], ...
               'Unrecognized Laplacian matrix choice.');
end




end
