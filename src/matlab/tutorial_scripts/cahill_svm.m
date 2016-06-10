function C = cahill_svm(XS, gt, numRows, numCols)
%{
Nathan D. Cahill Classification Report Scheme

%}

%% create training and testing data sets for classification
trainPrct = 0.10;
rng('default'); % so each script generates the same training/testing data
[trainMask,testMask,gtMask] = createTrainTestData(gt,trainPrct);
trainInd = find(trainMask);

%% predict labels using SVM classifier
labels = svmClassify(XS(trainInd,2:end),gt(trainInd),XS(:,2:end));

%% display ground truth and predicted label image
labelImg = reshape(labels,numRows,numCols);
figure; imshow(gt,[0 max(gt(:))]); title('Ground Truth Class Labels');
figure; imshow(gtMask); title('Ground Truth Mask');
figure; imshow(labelImg,[0 max(gt(:))]); title('Predicted Class Labels');
figure; imshow(labelImg.*gtMask,[0 max(gt(:))]); title('Predicted Class Labels in Ground Truth Pixels');

%% construct confusion matrix
C = confusionmat(gt(testMask&gtMask),labels(testMask&gtMask));

CMat = confusionmat(gt(testMask&gtMask),labels(testMask&gtMask));
[BC,R] = binaryClassificationResults(CMat);
OA = trace(CMat)/sum(CMat(:));
AA = nanmean(R(:,11));
APr = nanmean(R(:,12));
ASe = nanmean(R(:,13));
ASp = nanmean(R(:,14));


fprintf('Per-Class Accuracy\t\t\tUnc\n');
for i = 1:16
    fprintf('Class %02.0f:\t\t\t\t\t%6.4f\n',i,R(i,11));
end
fprintf('\nPer-Class Precision\t\t\tUncL\n');
for i = 1:16
    fprintf('Class %02.0f:\t\t\t\t\t%6.4f\n',i,R(i,12));
end
fprintf('\nPer-Class Sensitivity\t\tUnc\n');
for i = 1:16
    fprintf('Class %02.0f:\t\t\t\t\t%6.4f\n',i,R(i,13));
end
fprintf('\nPer-Class Specificity\t\tUnc\n');
for i = 1:16
    fprintf('Class %02.0f:\t\t\t\t\t%6.4f\n',i,R(i,14));
end

fprintf('\n\t\t\t\t\t\t\tUnc\n');
fprintf('Overall Accuracy:\t\t\t%6.4f\n',OA);
fprintf('Average Accuracy:\t\t\t%6.4f\n',AA);
fprintf('Average Precision:\t\t\t%6.4f\n',APr);
fprintf('Average Sensitivity:\t\t%6.4f\n',ASe);
fprintf('Average Specificity:\t\t%6.4f\n',ASp);

end