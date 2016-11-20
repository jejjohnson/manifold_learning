classdef Laplacian < handle
%==========================================================================
%
% LAPLACIAN is a class that computes the Laplacian matrix from an adjacency
% matrix. This class is initiated with an options structure and an
% adjacency matrix. Some key methods include computing different Laplacian
% matrices which are key as static methods to allow externel access without
% this specific class initiation.
%
% Examples
% --------
% 
% Properties
% ----------
% * W       - array, (N x N), adjacency matrix
% * lapType - str, type of Laplacian matrix to construct
%       + 'unnormalized'
%       + 'normalized'
%       + 'geometric' (TODO)
%       + 'randomwalk' (TODO)
%       + 'symmetric' (TODO)
%       + 'renormalized' (TODO)
% * matType - str, type of matrix to construct ['dense', 'sparse']
% 
%
% Methods
% -------
% * Laplacian   - CONSTRUCTOR, takes an adjacency matrix dataset and an
%                 optional struct with fieldnames for controllable
%                 parameters
% * getLaplacian - constructs the Laplacian matrix based on the properties
%                  set.
% * constructLap - static method which explicitly constructs the Laplacian
%                  matrix using the adjacency matrix 
% * displaplacian - displays the current Laplacian matrix
%
% Information
% -----------
% Author        : J. Emmanuel Johnson
% Email         : emanjohnson91@gmail.com
%               : jej2744@rit.edu
% Date          : 19th November, 2016
%
%==========================================================================
properties (Access = public)
    
    W           % Adjacency matrix
    L           % Laplacian matrix
    lapType     % type of Laplacian matrix
    D           % Degree matrix
    matType     % type of matrix (dense or sparse)
    
end

properties (Access = private)
    
    LapSettings     % Settings file for the Laplacian matrix construction
    degW            % Needed for normalized cases
    
end

methods (Access = public)
    
    % CONSTRUCTOR
    function self = Laplacian(adjacency, varargin)
        
        % check for input parameters
        narginchk(0,1);
        
        % Parse inputs for settings struct
        parseinputs(adjacency, varargin);
        
    end
    
    % GET LAPLACIAN MATRIX
    function L = getLaplacian(self)
        
        % Find the Laplacian matrix
        [L,self.D] = constructLap(self.W, self.LapSettings);
        
        % Save Laplacian, Degree matrix to class
        self.L = L;
        
    end
    
    % DISPLAY LAPLACIAN MATRIX
    function displaplacian(self)
        
        % check if Laplacian matrix has been calculate
        if isempty(self.L)
            self.L = getlaplacian(self);
        end
        
        % Plot the Laplacian matrix
        figure;
        spy(self.L);
        
    end
    
end

methods (Access = private)
    
    % PARSE INPUTS
    function parseinputs(self, data, InputArgs)
        %==================================================================
        %
        % PARSEINPUTS parses the InputArgs struct for the following inputs:
        % * lapType     - Laplacian matrix type
        % * matType     - matrix type
        %
        %==================================================================
        
        % Check for existence of Options
        if isempty(InputArgs)               % Check if empty
            % Create empty Options struct
            Options = struct;
        elseif ~isstruct(InputArgs{1})      % Check if type struct
            error([mfilename, 'parseInputs:badSettingsfile'], ...
                'Incorrect variable for Laplacian class.');
        else
            Options = InputArgs{1};
        end
        
        % Check if data is 2D, numeric
        classes = {'numeric'};
        attributes = {'2d'};
        validateattributes(data, classes, attributes);
        
        % Intiate inpute parse class
        p = inputParser;
        
        % Laplacian Matrix Type Validation
        paramName = 'lapType';
        params = {'unnormalized', 'normalized'};
        default = 'unnormalized';
        errorMsg = 'String must be "unnormalized", "normalized"';
        validationFcn = @(x) any(validatestring(x, params), errorMsg);
        addParameter(p, paramName, default, validationFcn);
        
        % Matrix Type validation
        paramName = 'matType';
        params = {'sparse', 'dense'};
        default = 'sparse';
        errorMsg = 'String must be "sparse", "dense".';
        validationFcn = @(x) any(validatestring(x, params), errorMsg);
        addParameter(p, paramName, default, validationFcn);
        
        % Parse struct
        parse(p, Options);
        
        % Set Parse Class properties to Class Instance Properties 
        self.LapSettings = p.Results;
        self.lapType = p.Results.lapType;
        self.matType = p.Results.matType;
        
    end
    
end

methods (Static)
    
    function [varargout] = constructlap(adjacency, varargin)
    
    % check number input arguments
    narginchk(0,1);
    
    % check number of output arguments
    nargoutchk(1,2);
    
    % parse inputs with subfunction
    Settings = parseinputs(adjacency, varargin);
    
    % Calculate Degree Matrix
    degW = sum(adjacency, 2);
    D = sparse(1:size(W,1), 1:size(W,2), degW);
    
    % Avoid dividing by zero
    degW(degW == 0) = eps;
    
    switch Settings.lapType
        
        case 'unnormalized'
            
            % Calculate the Unnormalized Laplacian Matrix
            L = D - W;
            
        case 'normalized'
            
            % Calculate the Unnormalized Laplacian Matrix
            L = D - W;
            
            % Calculate D^(-1/2)
            D = spdiags(1./(degW .^ 0.5), 0, size(D, 1), size(D, 2));
            
            % Calculate the normalized Laplacian
            L = D * L * D;
    end
    
    % Outputs
    switch nargout
        case 1
            varargout{1} = L;
        case 2
            varargout{1} = L;
            varargout{2} = D;
    end
    
    %-------------------------------%
    %-- SUBFUNCTION: Parse Inputs --%
    %-------------------------------%
    function Settings = parseinputs(adjacency, InputArgs)
        
        % Check for existence of Options
        if isempty(InputArgs)               % Check if empty
            % Create empty Options struct
            Options = struct;
        elseif ~isstruct(InputArgs{1})      % Check if type struct
            error([mfilename, 'parseInputs:badSettingsfile'], ...
                'Incorrect variable for Laplacian class.');
        else
            Options = InputArgs{1};
        end
        
        % Check if data is 2D, numeric
        classes = {'numeric'};
        attributes = {'2d'};
        validateattributes(adjacency, classes, attributes);
        
        % Intiate inpute parse class
        p = inputParser;
        
        % Laplacian Matrix Type Validation
        paramName = 'lapType';
        params = {'unnormalized', 'normalized'};
        default = 'unnormalized';
        errorMsg = 'String must be "unnormalized", "normalized"';
        validationFcn = @(x) any(validatestring(x, params), errorMsg);
        addParameter(p, paramName, default, validationFcn);
        
        % Matrix Type validation
        paramName = 'matType';
        params = {'sparse', 'dense'};
        default = 'sparse';
        errorMsg = 'String must be "sparse", "dense".';
        validationFcn = @(x) any(validatestring(x, params), errorMsg);
        addParameter(p, paramName, default, validationFcn);
        
        % Parse struct
        parse(p, Options);
        
        Settings = p.Results;

    end
        
    end
    
end

end