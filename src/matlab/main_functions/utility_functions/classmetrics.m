function [C, stats] = classmetrics(y_test, y_pred)

% construct the confusion matrix
C = confusionmat(y_test, y_pred);

% do the binary classification results from matrix
[~, R, stats.k, stats.v] = binaryClassificationResults(C);

% store appropriate statistics
stats.OA = trace(C)/sum(C(:));
stats.AA = nanmean(R(:,11));
stats.APr = nanmean(R(:,12));
stats.ASe = nanmean(R(:,13));
stats.ASp = nanmean(R(:,14));







end