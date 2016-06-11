function [trainMask,testMask,gtMask] = createTrainTestData(gtLabels,trainPrct)
% createTrainTestData: create boolean vectors indicating training data,
%   testing data, and available ground truth
% usage: [trainMask,testMask,gtMask] = createTrainTestData(gtLabels,trainPrct);
%
% arguments:
%   gtLabels (array or image) - ground truth class labels, valued 0 through
%       n. Any labels of 0 are assumed to be unknown.
%   trainPrct (scalar) - ratio of labeled class data to be selected for
%       training.
%   
%   trainMask, testMask, gtMask (same size as gtLabels) - boolean arrays
%       indicating whether current point has been selected as a training
%       datum, a testing datum, and/or has ground truth available
%

% author: Nathan D. Cahill
% email: nathan.cahill@rit.edu
% date: 5 July 2014

% ground truth mask
gtMask = (gtLabels~=0);

% unique ground truth class labels
uniqueLabels = unique(gtLabels(gtMask));

% loop through each class, selecting training data
trainMask = false(size(gtLabels));
for i = 1:numel(uniqueLabels)
    
    % get indices of points having current class label
    currentClass = find(gtLabels==uniqueLabels(i));
    
    % randomly shuffle them, then select a percentage for training
    ccShuffled = currentClass(randperm(numel(currentClass)));
    ccTrainInd = ccShuffled(1:ceil(trainPrct*numel(currentClass)));
    trainMask(ccTrainInd) = true;
    
end

% create test mask
testMask = ~trainMask;
