function [imgVec, imgDim] = imgtoarray(image)

% get dimensions of the image
[numRows,numCols,numSpectra] = size(image);

% convert the image into an image array
imgVec = reshape(image,[numRows*numCols numSpectra]);

% save dimensions
imgDim.numRows = numRows;
imgDim.numCols = numCols;
imgDim.numSpectra = numSpectra;


end