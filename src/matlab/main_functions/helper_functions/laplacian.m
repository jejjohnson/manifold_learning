function varargout = laplacian(W, varargin)
%{
======================================================================
LAPLACIAN computs the laplacian matrix of an adjacency matrix, W.

Examples
--------

Parameters
----------
* W   - array (N x N)
          an adjacency matrix representing graph G. Could be sparse
          or dense.
* Options - a MATLAB struct (optional)
      + lapType - str, 
            ~ 'unnormalized' (default)
            ~ 'normalized'
            ~ 'geometric'
            ~ 'randomwalk'
            ~ 'symmetric'
            ~ 'renormalized'
      + matType - str, ['dense', 'sparse']
Returns
-------


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
degreeMat = sum(W, 2);
D = sparse(1:size(W,1), 1:size(W,2), degreeMat);


%=====================================================================
%-- Laplacian Matrix --%

% Compute the Laplacian Matrix
switch laptype
    
    case 'unnormalized'




end
