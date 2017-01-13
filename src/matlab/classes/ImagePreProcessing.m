classdef ImagePreProcessing < handle
%==========================================================================
%
% IMAGEPREPROCESSING is a class that handles the preprocessing steps in for
% the images. It caters to hyperspectral image processing which enables the
% user to normalize the image as well as create an image vector for
% learning. There is also functionality to extract the spatial coordinates
% from the raster grid. Most of the methods are static and can be called
% without an instance of the object. However the constructor features the
% entire image processing chain with properties that feature the final
% image products.
%
% Examples
% --------
% 
% - this features a simple script that shows you how to obtain the final
% image products from the class.
% >> image = load('path-to-image');
% >> gt = load('path-to-groundtruth');
% >> ImageData = ImagePreProcessing(image, gt);
% >> imageVec = ImageData.imageVector;
% >> gtVec = ImageData.gtVector;
% >> spaData = ImageData.spatialData;
%
% Methods
% -------
% * imgvectorization    - transforms a 2D/3D image into a 1D/2D vector
%
% * imgnormalization    - normalizes the image between -1 and 1

% * getspatial          - gets the spatial coordinates (x,y) of an image
%
% TODO
% ----
% * TODO - write getvectorization method
% * TODO - write getnormalization method
% * TODO - write getspatial method
%
%==========================================================================

properties
    
    spatialData
    imageVector
    gtVector
    
end

methods
    
    % CONSTRUCTOR
    function self = ImagePreProcessing(image, imageGT)
    %======================================================================
    %
    % IMAGEPREPROCESSING 
    % 
    % TODO
    % ----
    % * TODO - check input values/parameters
    % * TODO - write unit tests
    %======================================================================
        
    % Normalize image
    imageTransformed = self.imgnormalization(image);

    % Vectorize Image
    self.imageVector = self.imgvectorization(imageTransformed);
    self.gtVector =self.imgvectorization(imageGT);

    % Get spatial coordinates
    self.spatialData = self.getspatial(self.imageVector);
        
    end
    
end

methods (Static)
    
    % IMAGE VECTORIZATION
    function imageVec = imgvectorization(image)
    %======================================================================
    % IMGVECTORIZATION converts the image from a rastergrid format to a 
    % vector format. The image can be in a 2D or 3D format to be converted
    % into a 1D or 2D vector respectively. This is a static method that can 
    % be called without an instances of the IMAGEPREPROCESSING class.
    %
    % Examples
    % --------
    % 
    % >> img = load('path-to-image');
    % >> imgVec = ImagePreProcessing.imgvectorization(img);
    %
    % Parameters
    % ----------
    % * image       - 2D (N x M) or 3D (N x M x D) array
    %                 image which will be vectorized.
    %
    % Returns
    % -------
    % * imageVec    - 1D (N * M x 1) or 2D (N * M x D) array
    %                 returned vectorized image
    %
    % Information
    % -----------
    % * Author      : J. Emmanuel Johnson
    % * Email       : emanjohnson91@gmail.com
    % * Date        : 11-Jan-2017
    %
    % * TODO - Parse inputs
    %
    %======================================================================
        
    % get the dimensions of image
    [numRows, numCols, numSpectra] = size(image);

    scfact = mean(reshape(...
        sqrt(sum(image.^2, 3)), numRows*numCols, 1));

    image = image ./ scfact;

    % Convert to vector
    imageVec = reshape(image, [numRows*numCols numSpectra]);
        
    end
    
    % IMAGE NORMALIZATION
    function imageTransformed = imgnormalization(image)
    %======================================================================
    %
    % IMGNORMALIZATION is a function that normalizes an input image
    % (matrix) between -1 and 1. This is a static method that can be called 
    % without an instances of the IMAGEPREPROCESSING class.
    %
    % Examples
    % --------
    % 
    % >> image = load('path-to-image');
    % >> imageNew = ImagePreProcessing.imgnormalization(image);
    %
    % Parameters
    % ----------
    % * image               - 2D (N x M) or 3D (N x M x D) array
    %                         image to be vectorized
    %
    % Returns
    % -------
    % * imageTransformed    - 1D (N * M x 1) or 2D (N * M x D) array
    %                         returned vectorized image
    %
    % Information
    % -----------
    % * Author      : J. Emmanuel Johnson
    % * Email       : emanjohnson91@gmail.com
    % * Date        : 11-Jan-2017
    %
    % Reference
    % ---------
    % * Author      : Nathan Cahill
    % * Email       : ndcahill@rit.edu
    % * Link        :
    %
    % ----
    % * TODO - Add Options struct to allow different normalization methods.
    % * TODO - Add link to reference code
    % * TODO - Parse inputs
    %
    %======================================================================
    
    % get the dimensions of image
    [numRows, numCols, ~] = size(image);
    
    % get the scaling parameter
    scfact = mean(reshape(...
        sqrt(sum(image.^2, 3)), numRows*numCols, 1));
    
    % normalize image by scaling parameter
    imageTransformed = image ./ scfact;
        
    end
    
    % GET SPATIAL COORDINATES
    function spatialData = getspatial(image)
    %======================================================================
    %
    % GETSPATIAL will extract the spatial locations of the data. This is a 
    % static method that can be called  without an instances of the 
    % IMAGEPREPROCESSING class.
    %
    % Example
    % -------
    % 
    % >> image = load('path-to-image');
    % >> spaData = ImagePreProcessing.getspatial(image);
    %
    % Parameters
    % ----------
    % * image               - 2D (N x M) or 3D (N x M x D) array
    %                         image to be vectorized
    %
    % Returns
    % -------
    % * spaData             - 1D (N * M x 1) or 2D (N * M x D) array
    %                         returned vectorized image
    %
    % Information
    % -----------
    % * Author      : J. Emmanuel Johnson
    % * Email       : emanjohnson91@gmail.com
    % * Date        : 11-Jan-2017
    %
    % Reference
    % ---------
    % * Author      : Nathan Cahill
    % * Email       : ndcahill@rit.edu
    % * Link        :
    %
    % ----
    % * TODO - Add Options struct to allow different normalization methods.
    % * TODO - Add link to reference code
    % * TODO - Parse inputs
    %
    %======================================================================
    
    % get the dimensions of image
    [numRows, numCols, ~] = size(image);
    
    % Create a meshgrid
    [x, y] = meshgrid(1:numCols, 1:numRows);
    
    % Extract the spatial data
    spatialData = [x(:), y(:)];
        
        
    end
    
end

end