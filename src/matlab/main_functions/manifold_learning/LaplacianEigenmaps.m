function [varargout] = LaplacianEigenmaps(data, options)
% LaplacianEigenmaps
%     [embedding, lambda] = LaplacianEigenmaps(data, options)
% 
%         
% Parameters
% ----------
% data - (N x d) matrix of samples x dimensions 
% 
% options - a matlab structure data structure with fields in options to be
%           set. 
% 
%     - knn   -   a matlab struct of knn options. See Adjacncy.m for more
%                 options.
%     - embedding - a matlab struct of graph embedding options. See 
%                   GraphEmbedding.m for more options
% 
%     - n_components      -   int, [default = 20]
%             the reduced dimension of the data. number of eigenvalues and
%             eigenvectors to find.
%     - constraint        -   str, [default = 'degree']
%             the constraint matrix for the eigenvalue decomposition step.
%             options include:
%                             *   'identity'     
%                             *   'degree'
%                             *   'k_scaling' (TODO)
%                             *   'dissimilarity' (TODO)
% 
% Returns
% -------
% embedding - (N x d) matrix of sample projection x number of components
%             the projected embedding of the eigenvalue decomposition
% 
% lambda - (d) column vector as the sorted eigenvalues
% 
% References
% ----------
% 
% Deng Cai's subspace
% 
% 
% Written by Juan Emmanuel Johnson


%==========================================================================
% Construct Adjacency Matrix
%==========================================================================

[W, ~] = Adjacency(data, options.knn);

%==========================================================================
% Compute Embedding
%==========================================================================

[embedding, lambda ] = GraphEmbedding(W, options.embedding); 

switch nargout
    case 1
        varargout{1} = embedding;
    case 2
        varargout{1} = embedding;
        varargout{2} = lambda;
        
    otherwise
        error('Improper number of varagout.');
end

end