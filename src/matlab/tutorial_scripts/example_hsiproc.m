% Clear workspace
clear; close all force; clc;

% Import Image
% add file path of where my data is located
% addpath('/Users/eman.johnson/Documents/Data/hsi_images')
% addpath('/media/eman/Emans HDD/Data/Images/RS/IndianPines')


% load the images
load('Indian_pines_corrected.mat');
load('Indian_pines_gt.mat');

% set them to variables
img = indian_pines_corrected;
gt = indian_pines_gt;

% clear the path as well as the datafiles 
clear indian*
% rmpath('/media/eman/Emans HDD/Data/Images/RS/IndianPines')
% rmpath('/Users/eman.johnson/Documents/Data/hsi_images')

%% Step-by-Step Image Processing

% Normalize image
img = ImagePreProcessing.imgnormalization(img);

% Vectorize Image
imageVec = ImagePreProcessing.imgvectorization(img);
gtVec = ImagePreProcessing.imgvectorization(gt);

% Get spatial coordinates
spaData = ImagePreProcessing.getspatial(img);

%% Complete Class Image Processing

% initialize class
ImageData = ImagePreProcessing(img, gt);
