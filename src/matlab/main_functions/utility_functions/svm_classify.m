function [y_pred, varargout] = svm_classify(...
    X_train, y_train, X_test, varargin)
% svmClassify used to perform classification using multiclass SVM.
%
% Function uses the One-versus-All (OVA) approach with the 
% standard MATLAB implementation. Returns the class labels based on
% the predicted max predicted probabilities. Also allows the option
% of using the model to predict an image. Returns predicted image.
%
% Example Usage
% -------------
%       [y_pred] = svmClassify(X_train, y_train, X_test);
%       [y_pred] = svmClassify(X_train, y_train, X_test, options);
%
% Parameters
% ----------
%   X_train     :   (N x D) matrix
%       N training data points of D features/dimensions
%
%   y_train     :   (N x 1) matrix
%       N samples with class labels for each training data point
%
%   X_test      :   (K x D) matrix
%       K testing data points of D features/dimensions
%
%   options     :   matlab struct with some options (optional)
%
%           * imgVec     :   (M x D) matrix, [default = nargin]
%               image (typically hyperspectral image) that the 
%               model can classify after the training data.
%               Note: D for imgVec does not necessarily have to be
%               the same size as D for X_train/X_test)
%
% Returns
% -------
%   testLabels  -   (Kx1) matrix
%       predicted class labels for each test data point
%   imgClass    -   (N x D) matrix of
%       particularly if one wants to classify an image from the model
%       of the training and testing data
%
% Information
% -----------
% Version:  v1
% Author:   Nathan D. Cahill
% Email:    nathan.cahill@rit.edu
% Date:     5 July 2014
% Version:  v2
% Author:   Juan Emmanuel Johnson
% email:    jej2744@rit.edu
% date:     12-Jun-16
% 
% TODO
% ----
% * Documentation
%       - References
%       - Examples 
% * Function Features
%       - options to save the SVM model matlab struct
%       - more kernel options (e.g. linear, polynomial, etc)
%       - probability plots
%       - One-versus-One (OVO) approach
%       


%==========================================================================
% parse inputs - parse inputs function
%==========================================================================

[X_train, y_train,X_test, options] = parseInputs(X_train, y_train, ...
    X_test, varargin);

%==========================================================================
% train the svm model on training data
%==========================================================================

% get unique labels and number of classes
uniqueLabels = unique(y_train);
numClasses = numel(uniqueLabels);

% train SVM for each class (using one-vs-all approach)
SVMModel = cell(numClasses,1);
for i = 1:numClasses
    currentClass = (y_train==uniqueLabels(i));
    SVMModel{i} = fitcsvm(X_train,currentClass,...
        'KernelFunction','rbf',...
        'Standardize',true,...
        'ClassNames',[false,true],...
        'KernelScale','auto');
end

%==========================================================================
% classify the test data
%==========================================================================

score = zeros(size(X_test,1),numClasses);
for i = 1:numClasses
    [~,tempScores] = predict(SVMModel{i},X_test);
    score(:,i) = tempScores(:,2);
end
[~,y_pred] = max(score,[],2);

%==========================================================================
% classify the image vector (optional)
%==========================================================================

if isfield(options, 'imgVec')
    
    score = zeros(size(options.imgVec,1), numClasses);
    for i = 1:numClasses
        [~, tempScores] = predict(SVMModel{i}, options.imgVec);
        score(:, i) = tempScores(:,2);
    end
    
    [~, varargout{1}] = max(score,[],2);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subfunction parseInputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [X_train,y_train,X_test, options] = parseInputs(...
    X_train, y_train, X_test, varargin)

%==========================================================================
% check dimensions and datatype of input variables
%==========================================================================

% training data
if ~isequal(ndims(X_train),2) || ~ismatrix(X_train)
    error([mfilename, ':parseInputs:baddX_trainData'],...
        'Training Data must be an N x D array.');
end

% training labels
if ~isvector(y_train) 
    error([mfilename, ':parseInputs:baddy_trainData'],...
        'Training Labels must be an N x 1 array.');
end

% testing data
if ~isequal(ndims(X_test),2) || ~ismatrix(X_test)
    error([mfilename, ':parseInputs:baddX_testData'],...
        'Testing Data must be an N x D array.');
end

%==========================================================================
% check input variables match each other appropriately
%==========================================================================

if ~isequal(size(X_train,1), size(y_train,1))
    error([mfilename, ':parseInputs:badX_trainy_traininputs'],...
        'Training Data and Testing Data samples must be size N');
end

if ~isequal(size(X_train,1), size(y_train,1))
    error([mfilename, ':parseInputs:badX_trainy_traininputs'],...
        'Training Data and Testing Data samples must be size N');
end

%==========================================================================
% check optional inputs
%==========================================================================

% if there is no variable input for options 
if isempty(varargin{1})
    options = 'None';
else
    temp = varargin{1};
    options = temp{1};
    % check input for validity
    if ~ismatrix(options.imgVec) || ~isequal(ndims(options.imgVec), 2)
        error([mfilename, ':parseInputs:badimageinput'],...
            'Image must be an M x D matrix.');
    end
end



end
end



