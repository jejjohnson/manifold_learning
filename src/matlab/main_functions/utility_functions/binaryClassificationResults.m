function [BC,R,k,v] = binaryClassificationResults(C)
% binaryClassificationResults: computes per-class results for a
%   multivariate classification problem
% usage: [BC,R,k,v] = binaryClassificationResults(C);
%
% arguments:
%   C (nxn) - confusion matrix from CONFUSIONMAT
%
%   BC (2x2xn) - binary confusion matrices for each class
%   R (nxm) - array of results. Rows are per-class results.
%       Columns are given by:
%           R(:,1) - class label
%           R(:,2) - total ground truth examples
%           R(:,3) - True Positives (TP)
%           R(:,4) - True Negatives (TN)
%           R(:,5) - False Positives (FP)
%           R(:,6) - False Negatives (FN)
%           R(:,7) - True Positive Rate (TPR)
%           R(:,8) - True Negative Rate (TNR)
%           R(:,9) - False Positive Rate (FPR)
%           R(:,10) - False Negative Rate (FNR)
%           R(:,11) - Accuracy ((TP+TN)/(TP+TN+FP+FN))
%           R(:,12) - Precision (TP/(TP+FP))
%           R(:,13) - Sensitivity (TP/(TP+FN))
%           R(:,14) - Specificity (TN/(TN+FP))
%   k (1x1) - Kappa coefficient
%   v (1x1) - Kappa variance
%

% author: Nathan D. Cahill
% email: nathan.cahill@rit.edu
% date: 15 March 2013

n = size(C,1);
if ~isequal(size(C),[n n])
    error('Input array must be nxn');
end

% compute binary confusion matrices
BC = zeros(2,2,n);
for i = 1:n
    BC(1,1,i) = C(i,i); % true positives
    BC(1,2,i) = sum(C(i,:)) - C(i,i); % false positives
    BC(2,1,i) = sum(C(:,i)) - C(i,i); % false negatives
    BC(2,2,i) = sum(C(:)) - BC(1,1,i) - BC(1,2,i) - BC(2,1,i); % true negatives
end

% total number of examples
numAll = sum(C(:));

% initialize R
R = zeros(n,14);

% class labels
R(:,1) = (0:n-1)';

% total class members in ground truth
R(:,2) = sum(C,2);

% True Positives (TP)
R(:,3) = squeeze(BC(1,1,:));

% True Negatives (TN)
R(:,4) = squeeze(BC(2,2,:));

% False Positives (FP)
R(:,5) = squeeze(BC(1,2,:));

% False Negatives (FN)
R(:,6) = squeeze(BC(2,1,:));

% True Positive Rate (TPR)
R(:,7) = R(:,3)./R(:,2);

% True Negative Rate (TNR)
R(:,8) = R(:,4)./(numAll-R(:,2));

% False Positive Rate (FPR)
R(:,9) = R(:,5)./R(:,2);

% False Negative Rate (FNR)
R(:,10) = R(:,6)./(numAll - R(:,2));

% Accuracy ((TP+TN)/(TP+TN+FP+FN))
R(:,11) = (R(:,3)+R(:,4))./(R(:,3)+R(:,4)+R(:,5)+R(:,6));

% Precision (TP/(TP+FP))
R(:,12) = R(:,3)./(R(:,3)+R(:,5));

% Sensitivity (TP/(TP+FN))
R(:,13) = R(:,3)./(R(:,3)+R(:,6));

% Specificity (TN/(TN+FP))
R(:,14) = R(:,4)./(R(:,4)+R(:,5));

% Kappa coefficient
Pa = trace(C)/numAll;
CcolSum = sum(C,1);
CrowSum = sum(C,2);
Pe = (CcolSum*CrowSum)/(numAll^2);
k = (Pa - Pe)/(1-Pe);

% Kappa variance
tempSum = 0;
for i = 1:n
    tempSum = tempSum + C(i,i)*(CrowSum(i)+CcolSum(i));
end
a1 = (1/numAll^2)*tempSum;

tempSum = 0;
for i = 1:n
    for j =1:n
        tempSum = tempSum + C(j,i)*(CrowSum(i)+CcolSum(j))^2;
    end
end
a2 = (1/numAll^3)*tempSum;

v = (1/numAll)*(Pa*(1-Pa)/(1-Pe)^2 + 2*(1-Pa)*(2*Pa*Pe-a1)/(1-Pe)^3 + ...
    (1-Pa)^2*(a2-4*Pe^2)/(1-Pe)^4);