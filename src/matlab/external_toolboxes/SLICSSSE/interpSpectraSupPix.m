function YI = interpSpectraSupPix(I,spSegs,meanPosition,spRad,Y)
%

% I - image (numRows x numCols x D)
% spSegs (numRows x numCols) indexed 0 to M-1
% meanSpectra (MxD)
% Y (MxK), where K<<D
% spRad (scalar) - radius of superpixels from which to interpolate

% YI (numRows x numCols x K)

[numRows,numCols,D] = size(I);
[M,K] = size(Y);

% initialize YI
YI = zeros(numRows,numCols,K);

% arrays for row, col pixel position
[r,c] = ndgrid(1:numRows,1:numCols);

% % determine cubic surface that interpolates Y values
% for i = 1:K
%     F = scatteredInterpolant(meanPosition(:,1),meanPosition(:,2),...
%         Y(:,i),'natural','linear');
%     YI(:,:,i) = F(r,c);
% end

% loop over each superpixel, interpolating all pixels inside the superpixel
for ii = 1:M

    % reset supepixel radius
    CurrentSpRad = spRad;
    
    % create mask for the i-1st superpixel
    mask = (spSegs==(ii-1));
    [ind1,ind2] = find(mask);
    numMaskPixels = sum(mask(:));
    
    % find neighboring superpixels
    spNeighbors = unique(spSegs(imdilate(mask,ones(3,3))));
    while CurrentSpRad>1
       
        % create mask of all neighboring superpixels
        newMask = mask;
        for jj = 1:numel(spNeighbors)
            newMask = newMask | (spSegs==spNeighbors(jj));
        end
        
        % update list of neighbors
        spNeighbors = unique(spSegs(imdilate(newMask,ones(3,3))));
        
        % decrement CurrentSpRad
        CurrentSpRad = CurrentSpRad - 1;
        
    end
   
    % grab values from I in the current superpixel
    testPosition = zeros(numMaskPixels,2);
    for jj = 1:numMaskPixels
       testPosition(jj,1) = r(ind1(jj),ind2(jj));
       testPosition(jj,2) = c(ind1(jj),ind2(jj));
    end
    
    % interpolate testSpectra
    S = interpolateSpectra(testPosition,meanPosition(spNeighbors+1,:),Y(spNeighbors+1,:));
    
    for jj = 1:numMaskPixels
        YI(ind1(jj),ind2(jj),:) = reshape(S(jj,:),[1,1,K]);
    end
end