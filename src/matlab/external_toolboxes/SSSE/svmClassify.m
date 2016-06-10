function testLabels = svmClassify(varargin)
% svmClassify - perform classification using multiclass SVM
% usage: testLabels = svmClassify(trainData,trainLabels,testData);
%
% arguments:
%   trainData (NxM) - N training data points in M-dimensional space
%   trainLabels (Nx1) - class labels for each training data point
%   testData (KxM) - K test data points in M-dimensional space
%   
%   testLabels (Kx1) - predicted class labels for each test data point
%

% author: Nathan D. Cahill
% email: nathan.cahill@rit.edu
% date: 5 July 2014

% parse inputs
[trainData,trainLabels,testData] = parseInputs(varargin{:});

% get unique labels and number of classes
uniqueLabels = unique(trainLabels);
numClasses = numel(uniqueLabels);

% train SVM for each class (using one-vs-all approach)
SVMModel = cell(numClasses,1);
for i = 1:numClasses
    currentClass = (trainLabels==uniqueLabels(i));
    SVMModel{i} = fitcsvm(trainData,currentClass,...
        'KernelFunction','rbf',...
        'Standardize',true,...
        'ClassNames',[false,true],...
        'KernelScale','auto');
end

% classify test data
score = zeros(size(testData,1),numClasses);
for i = 1:numClasses
    [~,tempScores] = predict(SVMModel{i},testData);
    score(:,i) = tempScores(:,2);
end
[~,testLabels] = max(score,[],2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% subfunction parseInputs
function [trainData,trainLabels,testData] = parseInputs(varargin)

% check number of inputs
narginchk(3,3);

% get/check training data
trainData = varargin{1};
if ~ismatrix(trainData)
    error([mfilename,':parseInputs:badTrainingData'],'Training data must be NxM array.');
end

% get/check training labels
trainLabels = varargin{2};
if ~isequal(size(trainLabels,1),size(trainData,1)) || ~isvector(trainLabels)
    error([mfilename,':parseInputs:badTrainingLabels'],'Training labels must be Nx1 array, where N = size(trainData,1).');
end

% get/check testing data
testData = varargin{3};
if ~ismatrix(testData) || ~isequal(size(testData,2),size(trainData,2))
    error([mfilename,':parseInputs:badTestingData'],'Testing data must be KxM array, where M = size(trainData,2).');
end
