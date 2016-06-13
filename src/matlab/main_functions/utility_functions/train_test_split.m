function [X_train, y_train, X_test, y_test, idx, masks] = ...
    train_test_split(X, y, options)
% train_test_splot used to output training and testing data and labels.
%
% Function reads in X data and y labels. Returns X_train and X_test 
% data, y_train and y_test labeled data and idx used. Also allows for 
% specified training percentage data which specifies the ratio of 
% labeled class data for training.
%
% Example Usage
% -------------
%       [X_train, y_train, X_test, y_test, idx, masks] = ...
%                   train_test_split(X, y, options)
%
% Parameters
% ----------
%   X           :   (N x D) matrix
%       N data points and D features/dimensions
%
%   y           :   (M x 1) matrix 
%       M samples with class labels with labeled data points
%       (note: assumes labels are > 0 and any 0 labeled data 
%       point will be considered to be unlabeled information)
%
%   options     :   matlab struct with more options listed below:
%   
%               * trainPrnt     :   float, [0, 1]
%                   ratio of labeled class data to be selected for
%                   training.
%
% Returns
% -------
% X_train           :   (K1 x D) matrix 
%       matrix of K1 training samples for D features
%
% y_train           :   (K1 x 1) matrix
%       vector of K1 labeled training samples
%
% X_test            :   (K2 x D) matrix
%       matrix of K2 testing samples for D features
%   
% y_test            :   (K2 x D) matrix
%       vector of K2 labeled testing samples
% 
% idx               :   matlab struct with the following fields:
%
%       * trainInd  :   training indices used
%       * testInd   :   testing indices used
%       * gtInd     :   available y indices for training/testing
% masks             :   matlab struct with the following fields:
%
%       * trainMask :   boolean mask of values for training
%       * testMask  :   boolean mask of values for training
%       * gtMask    :   boolean mask of values for groundtruth
%
% Information
% -----------
% Version   : v1
% Author    : Nathan D. Cahill
% Email     : nathan.cahill@rit.edu
% Date      : 5-Jul-14
% Version   : v2
% Author    : Juan Emmanuel Johnson
% Email     : jej2744@rit.edu
% Date      : 12-Jun-16
%
% TODO
% ----
% * Documentation
%       -   References
%       -   Examples
% * Function Features
%       -   Choose Testing data instead of training data
%       -   Use MATLABs inbuilt training vs. testing feature
%       -   Choose # of labeled samples instead of ratio
%       -   Stratified approachs
%

%==========================================================================
% parse inputs - parse inputs function
%==========================================================================

[X, y, options] = parseInputs(X, y, options);

%==========================================================================
% Pair X data points w/ y labels
%==========================================================================

% find out how many non-zero entries we have (i.e. how many training
% samples are available

% get a ground truth mask
gtMask = (y~=0);

% unique ground truth class labels
uniqueLabels = unique(y(gtMask));

% loop through each class, selecting training data
trainMask = false(size(y));
for i = 1:numel(uniqueLabels)
    
    % get indices of points having current class label
    currentClass = find(y==uniqueLabels(i));
    
    % randomly shuffle them, then select a percentage for training
    ccShuffled = currentClass(randperm(numel(currentClass)));
    ccTrainInd = ccShuffled(...
        1:ceil(options.trainPrct*numel(currentClass)));
    trainMask(ccTrainInd) = true;
    
end

% create test mask
testMask = ~trainMask;

% get training and testing Indices
trainInd = find(trainMask);
testInd = find(testMask&gtMask);
gtInd = find(gtMask);

% return the training and testing indices
X_train = X(trainInd,:); y_train = y(trainInd);
X_test = X(testInd,:); y_test = y(testInd);

% save the indices for later
idx.train = trainInd;
idx.test = testInd;
idx.gt = gtInd;

% save masks for later
masks.trainMask = trainMask;
masks.testMask = testMask;
masks.gtMask = gtMask;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subfunction parseInputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [X, y, options] = parseInputs(...
    X, y, options)

% Check X data points
if ~isequal(ndims(X),2) || ~ismatrix(X)
    error([mfilename, ':parseInputs:badXdata'], ...
        'X data must be an N x D array.')
end

% Check y data points
if ~isvector(y) 
    error([mfilename, ':parseInputs:badydata'],...
        'y labels must be an M x D array.')
end

%-------------------------
% check optional fields
%-------------------------

% check for trainprnt field
if ~isfield(options, 'trainPrnt')
    options.trainPrnt = .1;             % default 10% training samples
end

% check trainprnt is a float and is between 0 and 1
if ~isfloat(options.trainPrnt) || options.trainPrnt < 0 || ...
        options.trainPrnt >1
    error([mfilename, ':parseInputs:badtrainPrntdata'],...
        'trainprnt must be a float value between 0 and 1.')
end

end

end
