function [varargout] = SchroedingerEigenmaps(X, options)
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
%       * type - str 
%           ~ 'spaspec'
%           ~ 'specspa'
%           ~ 'pl'
%           ~ 'spaspecpl'
%           ~ 'specspapl'
% 
%       * spectral_nn   - a matlab struct of knn options for spectral case. 
%           See Adjacncy.m for more options.
%
%       * spatial_nn    - a matlab struct of knn options for spatial case. 
%           See Adjacncy.m for more options.
%
%       * embedding - a matlab struct of graph embedding options. See 
%                   GraphEmbedding.m for more options
%
%       * partiallabels  - a matlab struct of partial labels options.
%           See PartialLabelsPotential.m for more options
%
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
    
    case 'spaspec'
        

        % get spatial positions of the image data
        [x,y] = meshgrid(1:size(options.ss.image,2),...
            1:size(options.ss.image,1));
        pData = [x(:) y(:)];

        % construct the spectral adjacency matrix
        [W, ~] = Adjacency(X, options.spectral_nn);

        % construct the spectral adjacency matrix
        [~, idxP] = Adjacency(pData, options.spatial_nn);
        
        % assign the cluster data to the spatial locations
        options.ss.clusterdata = pData;
        
        % assign the weighted data to the spectral locations
        options.ss.weightdata = X;
        
        % use the Knn values from the spatial data
        options.ss.idx = idxP;
        
        % declare spatial-spectral graph embedding
        options.embedding.type = 'ss';
        options.embedding.ss = options.ss;

    case 'specspa'
        
        % get spatial positions of the image data
        [x,y] = meshgrid(1:size(options.ss.image,2),...
            1:size(options.ss.image,1));
        pData = [x(:) y(:)];
        
        % construct the spectral adjacency matrix
        [W, idxF] = Adjacency(X, options.spectral_nn);
        
        % assign the cluster data to the spectral locations
        options.ss.clusterdata = X;
        
        % assign the weighted data to the spatial locations
        options.ss.weightdata = pData;
        
        % use the Knn values from the spatial data
        options.ss.idx = idxF;
        
        % declare spatial-spectral graph embedding
        options.embedding.type = 'ss';
        options.embedding.ss = options.ss;
        
    case 'pl'
        
        % declare partial-labels graph embedding
        options.embedding.type = 'pl';
        
        % construct the spectral adjacency matrix
        [W, ~] = Adjacency(X, options.spectral_nn);
        
        % decide if constraint conditioning is true or not
        if isfield(options, 'constraintConditioning')
            options.embedding.pl.constraintConditioning = true;
        end
        
        % declare partial-labels graph embedding
        options.embedding.type = 'pl';
        options.embedding.pl = options.pl;
        
    case 'spaspecpl'
        
        % get spatial positions of the image data
        [x,y] = meshgrid(1:size(options.ss.image,2),...
            1:size(options.ss.image,1));
        pData = [x(:) y(:)];

        % construct the spectral adjacency matrix
        [W, ~] = Adjacency(X, options.spectral_nn);

        % construct the spectral adjacency matrix
        [~, idxP] = Adjacency(pData, options.spatial_nn);
        
        % assign the cluster data to the spatial locations
        options.ss.clusterdata = pData;
        
        % assign the weighted data to the spectral locations
        options.ss.weightdata = X;
        
        % use the Knn values from the spatial data
        options.ss.idx = idxP;
        
        % declare spatial-spectral graph embedding
        options.embedding.ss = options.ss;
        
        % declare partial-labels graph embedding
        options.embedding.type = 'pl';
        
        % decide if constraint conditioning is true or not
        if isfield(options, 'constraintConditioning')
            options.embedding.pl.constraintConditioning = true;
        end
        
        % declare partial-labels graph embedding
        options.embedding.pl = options.pl;
        
        % declare partial-labels graph embedding
        options.embedding.type = 'sspl';
        
    case 'specspapl'
        
        % get spatial positions of the image data
        [x,y] = meshgrid(1:size(options.ss.image,2),...
            1:size(options.ss.image,1));
        pData = [x(:) y(:)];
        
        % construct the spectral adjacency matrix
        [W, idxF] = Adjacency(X, options.spectral_nn);
        
        % assign the cluster data to the spectral locations
        options.ss.clusterdata = X;
        
        % assign the weighted data to the spatial locations
        options.ss.weightdata = pData;
        
        % use the Knn values from the spatial data
        options.ss.idx = idxF;
        
        % declare spatial-spectral graph embedding
        options.embedding.ss = options.ss;
        
        % decide if constraint conditioning is true or not
        if isfield(options, 'constraintConditioning')
            options.embedding.pl.constraintConditioning = true;
        end
        
        % declare partial-labels graph embedding
        options.embedding.pl = options.pl;
        
        % declare spectral spatial partial labels
        options.embedding.type = 'sspl';

    otherwise
        error('Need a Schroedinger Operator type.')
        

end
        
       
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Subfunction - parseInputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [X, options] = parseInputs(X, options)
    
%--------------------------------------------------------------------------
% check mandatory inputs
%--------------------------------------------------------------------------
if ~isequal(ndims(X),2)
    error([mfilename, ':parseInputs:badXdata'], ...
                'data must be 2-D double precision array');
end

%--------------------------------------------------------------------------
% check mandatory options
%--------------------------------------------------------------------------
    
% if no type for schroedinger eigenmap
if ~isfield(options, 'type')
    options.type = 'se';
end

% if no type spectral nn inputs
if ~isfield(options, 'spectral_nn')
    default_options = [];
    default_options.nn_graph = 'knn';
    default_options.k = 20;
    default_options.type = 'standard';
    default_options.saved = 0;
    options.spectral_nn = default_options;
end

% if no type spatial nn inputs
if ~isfield(options, 'spatial_nn')
    default_options = [];
    default_options.nn_graph = 'knn';
    default_options.k = 4;
    default_options.type = 'standard';
    default_options.saved = 0;
    options.spatial_nn = default_options;
end

% if no embedding inputs
if ~isfield(options, 'embedding')
    default_options = [];
    default_options.n_components = 150;
    default_options.constraint = 'degree';
    options.embedding = default_options;
end

%--------------------------------------------------------------------------
% default options 
%--------------------------------------------------------------------------

% options by type
switch lower(options.type)
    
    % spatial-spectral implementation
    
    case {'spaspec', 'specspa'}
        
        % check spatial-spectral cluster method
        if ~isfield(options, 'ssmethod')
            options.ssmethod = 'spaspec';
        end
        
        % use struct from options given
        ss_options = options.ss;
        
        % check that we have an image to do spatial knn for
        if ~isfield(ss_options, 'image')
            error([mfilename, ':parseInputs:noimgdata'], ...
                'must have an image file to do spatial locations');
        elseif ~isequal(ndims(ss_options.image),3)
            error([mfilename, ':parseInputs:badimgdata'],...
                'img must be 3-D array with double inputs.');
        end
        
        % check spatial parameters
        if ~isfield(options.spatial_nn, 'k')
            options.spatial_nn.k = 4;
        end
        
        % check spatial saved files
        if ~isfield(options.spatial_nn, 'saved')
            options.spatial_nn.saved = 0;
        end
        
        % trade off parameter alpha
        if ~isfield(ss_options, 'alpha')
            ss_options.alpha = 17.78;
        elseif ~isfloat(ss_options.alpha) || ss_options.alpha < 0
            error([mfilename, ':parseInputs:badalphadata'],...
                'alpha must be a float greater than 0.');
        end
        
        % cluster kernel for spatial data
        if ~isfield(ss_options, 'clusterkernel')
            ss_options.clusterkernel = 'heat';       
        elseif ~isequal(ss_options.clusterkernel, 'heat') || ...
            ~isequal(ss_options.clusterkernel, 'angle')
            error([mfilename, ':parseInputs:badclusterkerneldata'],...
                'clusterkernel must either "heat" or "angle".');
        end
        
        % sigma value for spatial
        if ~isfield(ss_options, 'spatialsigma')
            ss_options.spatialsigma = 1.0;
        elseif ~isfloat(ss_options.spatialsigma) || ...
                ss_options.spatialsigma < 0
            error([mfilename, ':parseInputs:badspatialsigmadata'],...
                'spatialsigma data must be a float greater than 0.');
        end
        
        % sigma value for spectral
        if ~isfield(ss_options, 'weightsigma')
            ss_options.weightsigma = 1.0;
        elseif ~isfloat(ss_options.weightsigma) || ...
                ss_options.weightsigma < 0
            error([mfilename, ':parseInputs:badweightsigmadata'],...
                'spatialsigma data must be a float greater than 0.');
        end
        
        % put struct back
        options.ss = [];
        options.ss = ss_options;
        
    case 'pl'
        
        % check mandatory data
        if ~isfield(options.pl, 'labels')
            error([mfilename, ':parseInputs:nolabels'], ...
                'Need some labels to do partial labels.');
        end
        
        % use struct from options given
        pl_options = options.pl;
        
        % check partial labels data types and size
        if ~isvector(pl_options.labels)
            error([mfilename, ':parseInputs:badydata'], ...
                'labels must be a vector with double inputs.');
        end
        
        % check active constraints
        if ~isfield(pl_options, 'mlactive')
            pl_options.mlactive = ...
                1:numel(unique(options.pl.labels(options.pl.labels>0)));
        elseif ~isvector(pl_options.mlactive) || ...
                numel(pl_options.mlactive) > ...
                numel(unique(pl_options.labels(pl_options.labels>0))) ||...
                nume(pl_options.mlactive) < 0
            error([mfilename, ':parseInputs:badmlactivedata'],...
                'mlactive must be a vector of size 0 to # of classes.');
        end
        
        % trade off parameter for ss potential
        if ~isfield(pl_options, 'beta')
            options.beta = 17.78;
        elseif ~isfloat(pl_options.beta) || pl_options.beta < 0
            error([mfilename, ':parseInputs:badbetadata'],...
                'beta value must be a float greater than 0.');
        end
        
        % put struct back
        options.pl.pl = [];
        options.embedding.pl = pl_options;
        
end
        
end
        
        

end

