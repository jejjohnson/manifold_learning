classdef Adjacency < handle
%==========================================================================
% ADJACENCY is classdef 
%
% Properties
% ----------
%
% Methods
% -------
% 
% 
% Information
% -----------
% * Author  : J. Emmanuel Johnson
% * Email   : emanjohnson91@gmail.com
% * Date    : Sat. 5th Nov, 2016
%
%==========================================================================    
    properties (Access = public)
        
        kernelType = 'standard';
        nnGraph = 'knn';
        kNeighbors = 20;
        distance = 'euclidean';
        eRadius = 10;
        kernel = 'heat';
        sigma = 1.0;
        nnTime = 0;
        
    end
    
    properties (SetAccess = public, GetAccess = private)
        
        savedData = None;
        
    end
    
    methods
        
        % CONSTRUCTOR
        function self = Adjacency(Data, Settings)
            
            % 
            
            
        end
        
        % DESTRUCTOR
        
        
    end
    
    methods (Access = private)
        
        % Parse Inputs
        function self = parseinputs(self, data, Settings)
            
            
        end
        
        % Nearest Neighbor Search
        function [idx, dist] = nnSearch(self, data)
            
            % query the 
            
            
        end
        
        % KD Tree Search
        function [idx, dist] = kdSearch(self, data)
            
            tic;
            
            % initialize the KD Tree Model
            KDModel = KDTreeSearcher(data, ...
                'Distance', self.distance);
            
            % query the vector for k nearest neighbors
            [idx, dist] = knnsearch(KDModel, data, ...
                'k', self.kNeighbors+1);
            
            % discard the first distance
            idx = idx(:, 2:end);
            dist = dist(:, 2:end);
            self.nnTime = toc;
            
        end
        
        % Compute Weights
        function w = distfunction(self, dist)
            
            switch self.type
                
                case 'heat'
                    
                    w = exp(-(dist.^2) ./ (self.sigma .^2));
                    
                    
                case 'cosine'
                    
                    w = exp(-acos(1-dist));
                    
                otherwise 
                    error([mfilename, 'kernel:bakernel'], ...
                        'Unrecognized kernel function.');
            end
            
        end
        
        % Construct Adjacency Matrix
        function W = constructadjacency(self, data, idx, w)
            
            switch self.type
                
                case 'standard'
                    
                    % construct a sparse adjacency matrix
                    W = sparse(repmat((1:size(data, 1))', [1 self.x]), ...
                        idx, w, size(data, 1), size(data, 1));
                    
                    % make the matrix symmetric
                    W = max(W, W');
                    
                case 'similarity'
                    
                    Ws_left = sparse(repmat(data, 1, length(data)));
                    
                    Ws_right = sparse(Ws_left');
                    
                    W = Ws_left == Ws_right;
                    
                    % set all the zero values to be zero
                    W(data == 0, :) = 0;
                    W(:, data == 0) = 0;
                    
                    % double precious
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
                    
                otherwise
                    error([mfilename, 'constructadjacency:badtype'], ...
                        'Unrecognized adjacency matrix.');
            end
            
            
        end
        
    end
    
end