function [W, idx] = Adjacency(data, options)
%{
Parameters
----------

options - matlab struc


%}

%==========================================================================
% Check Inputs
%==========================================================================
[data, options] = parseInputs(data, options);

%==========================================================================
% Find the knn distances
%==========================================================================
    
switch options.type

    case 'standard'
        
        if options.saved == 1
            try
                load('saved_data/dist_standard')
                disp('previous spectral dist values found..')
            catch
                
                disp('previous spectral dist values not found..')
                disp('computing knn...')
                tic;
                % find the k nearest neighbors
                [idx, dist] = knnsearch(data, data,'k', ...
                    options.k+1, 'Distance', options.distance);

                % discard the first distance
                idx = idx(:, 2:end); dist = dist(:, 2:end);
                time = toc;

                % print the time taken
                fprintf('Knn Search: %.3f.s\n', time)

                % save data for later
                disp('saving spectral distance values for later...')
                save('saved_data/dist_standard', 'idx', 'dist', 'time') 
            end
        elseif options.saved == 2
            try
                load('saved_data/dist_spatial')
                disp('previous spatial distance values found...')
            catch
                disp('previous spatial distance values not found...')
                disp('computing knn distance')
                tic;
                % find the k nearest neighbors
                [idx, dist] = knnsearch(data, data,'k', ...
                    options.k+1, 'Distance', options.distance);

                % discard the first distance
                idx = idx(:, 2:end); dist = dist(:, 2:end);
                time = toc;

                % print the time taken
                fprintf('Knn Search: %.3f.s\n', time)

                % save data for later
                disp('saving spatial distance values for later...')
                save('saved_data/dist_spatial', 'idx', 'dist', 'time') 
            end
        else
            tic;
            % find the k nearest neighbors
            [idx, dist] = knnsearch(data, data,'k', ...
                options.k+1, 'Distance', options.distance);

            % discard the first distance
            idx = idx(:, 2:end); dist = dist(:, 2:end);
            time = toc;

            % print the time taken
            fprintf('Knn Search: %.3f.s\n', time)

        end


end


%==========================================================================
% Compute Weights
%========================================================================== 

switch options.type
    
    case 'standard'
        switch options.kernel

            % standard gaussian heat kernel
            case 'heat'

                w = exp(-(dist.^2)./(options.sigma.^2)); 

            % cosine angle kernel
            case 'cosine'

                w = exp(-acos(1-dist));

        end
end


%==========================================================================
% Construct Adjacency Matrix
%==========================================================================

switch options.type
    
    case 'standard'
        
        % construct a sparse adjacency matrix
        W = sparse(repmat((1:size(data,1))', [1 options.k]),...
            idx, w, size(data,1), size(data,1));
        
        % make the matrix symmetric
        W = max(W, W');
        
    case 'similarity'
        
        Ws_left = sparse(repmat(data,1,length(data)));
        
        Ws_right = sparse(Ws_left');
        
        W = Ws_left == Ws_right;
        
        % set all the zero values to be zero
        W(data == 0,:) = 0; 
        W(:,data == 0) = 0; 
        
        % double precision
        W = double(W);
        
    case 'dissimilarity'
        
        Ws_left = sparse(repmat(data,1,length(data)));
        
        Ws_right = sparse(Ws_left');
        
        W = Ws_left ~= Ws_right;
        
        figure;
        spy(W)
        % set all the zero values to be zero
        W(data == 0,:) = 0; 
        W(:,data == 0) = 0; 
        
        % double precision
        W = double(W);
        
        
end
        

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Subfunction - parseInputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [data, options] = parseInputs(data, options)


% if no type
if ~isfield(options, 'type')
    options.type = 'standard';
end
    


% check entry matrix depending upon the type
switch lower(options.type)
    
    case 'standard'
        
        if ~isequal(ndims(data),2) || ~isa(data, 'double')
            error([mfilename, ':parseInputs:baddata'], ...
                'data must be 2-D double precision array');
        end
        
    case 'similarity'
        
        if ~isequal(size(data,2), 1) || ~isa(data, 'double')
            
            error([mfilename, ':parseInputs:baddata'],...
                'data must be 1-D double array');
        end
    case 'dissimilarity'
        
        if ~isequal(size(data,2), 1) || ~isa(data, 'double')
            
            error([mfilename, ':parseInputs:baddata'],...
                'data must be 1-D double array');
        end
        
end
        
        

%--------------------------------------------------------------------------
% default options 
%--------------------------------------------------------------------------

% check the type
switch lower(options.type)
    
    % do the standard case first
    case 'standard'
        
        % default nn_graph method
        if ~isfield(options, 'nn_graph')
            options.nn_graph = 'knn';
        end
        
        % number of k values
        if ~isfield(options, 'k')
            options.k = 20;
        end
        
        % distance value for knn graph
        if ~isfield(options, 'distance')
            options.distance = 'euclidean';
        end
        
        % kernel used
        if ~isfield(options, 'kernel')
            options.kernel = 'heat';
        end
        
        % sigma parameter for kernel
        if ~isfield(options, 'sigma')
            options.sigma = 1.0;
        end
        
    
        
end
        

        
end
        
    
end
