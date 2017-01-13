function ImageData = getexampleimg

% load the images
load('Indian_pines_corrected.mat');
load('Indian_pines_gt.mat');

% set them to variables
img = indian_pines_corrected;
gt = indian_pines_gt;

% Complete Class Image Processing

% initialize class
ImageData = ImagePreProcessing(img, gt);

end

% Get example image from MATLAB image database
