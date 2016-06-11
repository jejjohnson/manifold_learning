function S = interpolateSpectra(testPosition,supPixPosition,supPixVals)
% interpolateSpectra: interpolate spectral data according to nearby
%   superpixels
% usage: S = interpolateSpectra(testSpectra,supPixSpectra,supPixVals)
%
% arguments:
%   testPosition (Kx2) - spatial coordinates of each of K points to be 
%       interpolated
%   supPixPosition (Mx2) - 2-dimensional mean spectral values for each of M
%       superpixels
%   supPixValues (MxN) - N-dimensional values at each superpixel
%
%   S (KxN) - interpolated N-dimensional values at each of the K test
%       points
%

% author: N. Cahill & S. Chew
% date: 7/30/2014

K = size(testPosition,1);
M = size(supPixPosition,1);
N = size(supPixVals,2);

% % compute distance matrix between test positions and superpixel positions
% fullTestPosition = repmat(permute(testPosition,[1 3 2]),[1 M 1]);
% fullsupPixPosition = repmat(permute(supPixPosition',[3 2 1]),[K 1 1]);
% distMat = sqrt(sum((fullTestPosition - fullsupPixPosition).^2,3));
% 
% % test to see if any distances are zero
% mask = distMat<eps;
% 
% % May need to revisit changing points with zero distance later
% %[zi,zj] = find(mask); 
% 
% % temporarily set zero distances to eps
% distMat(mask) = eps;
% 
% % inverse distance
% p = 0.1;
% invDistMat = 1./(distMat.^p);
% 
% % denominator of interpolation weight
% wsum = sum(invDistMat,2);
% S = (invDistMat*supPixVals)./repmat(wsum,[1,N]);

% griddata method
S = zeros(K,N);
for i = 1:N
    F = scatteredInterpolant(supPixPosition(:,1),supPixPosition(:,2),...
        supPixVals(:,i),'natural','linear');
    S(:,i) = F(testPosition(:,1),testPosition(:,2));
end