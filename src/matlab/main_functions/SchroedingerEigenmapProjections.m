function [embedding, lambda] = SchroedingerEigenmapProjections(...
    X, options)
% SchroedingerEigenmaps
%     [embedding, lambda] = SchroedingerEigenmaps(data, options)
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
% TODO
% ----
% * Do more checks for faulty inputs to prevent user frustration
%
%
% Information
% -----------
% Author: Juan Emmanuel Johnson
% Email: jej2744@rit.edu
% Date: 11 June 2016

%==========================================================================
% Parse Inputs
%==========================================================================
[X, options] = parseInputs(X, options);

%==========================================================================
% Schroedinger Eigenmaps Problem Tuner
%==========================================================================


% cases for the spatial-spectral method
switch lower(options.type)
    
    case 'spatialspectral'
        
        % get spatial positions of the image data
        [x,y] = meshgrid(1:size(options.image,2),1:size(options.image,1));
        pData = [x(:) y(:)];
        
        % construct the spectral adjacency matrix
        [W, ~] = Adjacency(X, options.knn);
        
        % construct the spectral adjacency matrix
        [~, idxP] = Adjacency(pData, options.spatial_nn);
        
        
        % assign the cluster data to the spatial locations
        options.embedding.clusterdata = pData;
        
        % assign the weighted data to the spectral locations
        options.embedding.weightdata = X;
        
        % use the Knn values from the spatial data
        options.embedding.idx = idxP;
        
        % declare spatial-spectral graph embedding
        options.embedding.type = 'ssse';

    case 'spectralspatial'
        
        % get spatial positions of the image data
        [x,y] = meshgrid(1:size(options.image,2),1:size(options.image,1));
        pData = [x(:) y(:)];
        
        % construct the spectral adjacency matrix
        [W, idxF] = Adjacency(X, options.knn);
        
        % assign the cluster data to the spectral locations
        options.embedding.clusterdata = X;
        
        % assign the weighted data to the spatial locations
        options.embedding.weightdata = pData;
        
        % use the Knn values from the spatial data
        options.embedding.idx = idxF;
        
        % declare spatial-spectral graph embedding
        options.embedding.type = 'ssse';
end
        
        

%==========================================================================
% Compute Embedding
%==========================================================================

[embedding, lambda ] = linear_graph_embedding(W, X, options.embedding); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Subfunction - parseInputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [X, options] = parseInputs(X, options)

    
% check X data inputs
if ~isequal(ndims(X),2)
    error([mfilename, ':parseInputs:badXdata'], ...
                'data must be 2-D double precision array');
end

%--------------------------------------------------------------------------
% check default types
%--------------------------------------------------------------------------

% if no type
if ~isfield(options, 'type')
    options.type = 'se';
end

% if no type
if ~isfield(options, 'spatial_nn')
    temp_options.k = 4;
    temp_options.type = 'standard';
    temp_options.saved = 0;
    options.spatial_nn = temp_options;
end

%--------------------------------------------------------------------------
% default options 
%--------------------------------------------------------------------------

% options by type
switch lower(options.type)
    
    % spatial-spectral implementation
    
    case 'spatialspectral'
        
        % check that we have an image to do spatial knn for
        if ~isfield(options, 'image')
            error([mfilename, ':parseInputs:nodata'], ...
                'must have an image file to do spatial locations');
        end
        
        % check spatial-spectral cluster method
        if ~isfield(options, 'ssmethod')
            options.ssmethod = 'spatialspectral';
        end
        
        % check spatial parameters
        if ~isfield(options.spatial_nn, 'k')
            options.spatial_nn.k = 4;
        end
        
        % check spatial saved files
        if ~isfield(options.spatial_nn, 'saved')
            options.spatial_nn.saved = 0;
        end
        
        % trade off parameter
        if ~isfield(options.embedding, 'alpha')
            options.embedding.alpha = 17.78;
        end
        
        % kernel for spatial data
        if ~isfield(options.embedding, 'clusterkernel')
            options.embedding.clusterkernel = 1;       % default heat kernel
        end
        
        % sigma value for spatial
        if ~isfield(options.embedding, 'spatialsigma')
            options.embedding.spatialsigma = 1.0;
        end
        
        % sigma value for spectral
        if ~isfield(options.embedding, 'weightsigma')
            options.embedding.weightsigma = 1.0;
        end
    case 'spectralspatial'
        
        % check that we have an image to do spatial knn for
        if ~isfield(option, 'image')
            error([mfilename, ':parseInputs:nodata'], ...
                'must have an image file to do spatial locations');
        end
        
        % check spatial-spectral cluster method
        if ~isfield(options, 'ssmethod')
            options.ssmethod = 'spatialspectral';
        end
        
        % check spatial parameters
        if ~isfield(options, 'spatial_nn')
            options.spatial_nn = 4;
        end
        
        % trade off parameter
        if ~isfield(options.embedding, 'alpha')
            options.embedding.alpha = 17.78;
        end
        
        % kernel for spatial data
        if ~isfield(options.embedding, 'clusterkernel')
            options.embedding.clusterkernel = True;       % default heat kernel
        end
        
        % sigma value for spatial
        if ~isfield(options.embedding, 'spatialsigma')
            options.embedding.clustersigma = 1.0;
        end
        
        % sigma value for spectral
        if ~isfield(options.embedding, 'weightsigma')
        end
        
end
        
end
        
        
        
        
        
        
        
    


end

