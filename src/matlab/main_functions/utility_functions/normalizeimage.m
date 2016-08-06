function image = normalizeimage(image)

% get dimensions of the image
[numRows,numCols,~] = size(image);

% normalize the image
scfact = mean(reshape(sqrt(sum(image.^2,3)),numRows*numCols,1));
image = image./scfact;
    
end