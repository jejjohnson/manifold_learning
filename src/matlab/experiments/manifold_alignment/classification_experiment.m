% Experiment Script


% Clear Variables
clear all; close all; clc;

% dataset
dataset = 'vcu';
img2 = 2;
algo = 'ssma';

switch lower(dataset)
    
    case 'vcu'
        
        % Import Images
        load H:\Data\Images\RS\VCU\vcu_images.mat;
        
        ImageData = [];
        ImageData{1}.img = vcu_data(1).img;             % 400m image
        ImageData{2}.img = vcu_data(2).img;             % 2000m image

        ImageData{1}.gt = vcu_data(1).img_gt2;      % 400m image ground truth
        ImageData{2}.gt = vcu_data(2).img_gt2;      % 2000m image ground truth
        
        
        % Image preprocessing
        for iImage = 1:numel(ImageData)

            % image preprocessing
            ImageData{iImage}.img = normalizeimage(ImageData{iImage}.img ); 

            % convert image to array
            [ImageData{iImage}.imgVec, ImageData{iImage}.dims] = imgtoarray(ImageData{iImage}.img);
    
            % convert ground truth to array
            [ImageData{iImage}.gtVec, ImageData{iImage}.gtdims] = imgtoarray(ImageData{iImage}.gt);
        end
        
        % Get Data in appropriate form
        Options.trainPrct = {.1, .1};
        Options.labelPrct = {.1, .1};
        Data = getdataformat(ImageData, Options);
        
    case 'lidar'
        
        try 
            display('Trying to load in previous data..');
            x
            load H:\Data\Images\RS\lidar_hsi\Main\Data.mat
        catch
            display('Failed...');
            display('No file found.');
            display('Loading in data manually. May take a while...');
            
            
            filePath = 'H:\Data\Images\RS\lidar_hsi\Main\';

            ImageData = [];             % intialize image data structure
            Image1= imread([filePath, 'Eagle.tif']);
            Image2 = imread([filePath, 'lidar_intensity.tif']);
            Image3 = imread([filePath, 'lidar_dsm.tif']);


            load H:\Data\Images\RS\lidar_hsi\Main\test
            load H:\Data\Images\RS\lidar_hsi\Main\train

            imgGT = train(:,:,1) + test(:, :, 1);
            % change type
            Image1 = double(Image1);
            Image2 = double(Image2);
            Image3 = double(Image3);
            display('Done!')

            % Normalize the images
            Image1 = normalizeimage(Image1);
            Image2 = normalizeimage(Image2);
            Image3 = normalizeimage(Image3);

            % Save the data
            save H:\Data\Images\RS\lidar_hsi\Main\image1.mat Image1 -v7.3
            save H:\Data\Images\RS\lidar_hsi\Main\image2.mat Image2 -v7.3
            save H:\Data\Images\RS\lidar_hsi\Main\image3.mat Image3 -v7.3
            save H:\Data\Images\RS\lidar_hsi\Main\imggt.mat imgGT -v7.3
            
            display('Done!')
            display('Clearing work space.')
            
            clear; close all; clc;
            
            display('Loading in data');
            
            Image1= load('H:\Data\Images\RS\lidar_hsi\Main\image1.mat');
            
            switch img2
                case 1
                    Image2 = load('H:\Data\Images\RS\lidar_hsi\Main\image2.mat');
                case 2
                    Image3 = load('H:\Data\Images\RS\lidar_hsi\Main\image3.mat');
                    Image2 = Image3;
                    clear Image3;
                otherwise
                    error('Unrecgonized image 2');
            end
            
            ImageGT= matfile('H:\Data\Images\RS\lidar_hsi\Main\imggt.mat');
            
            
            % Tune data
            ImageData = [];
            ImageData{1}.img = Image1;             % 400m image
            clear Image1;
            ImageData{2}.img = Image2;             % 2000m image
            clear Image2;

            ImageData{1}.gt = ImageGT;      % 400m image ground truth
            ImageData{2}.gt = ImageGT;      % 2000m image ground truth
            clear ImageGT;


            % Image preprocessing
            for iImage = 1:numel(ImageData)

                % image preprocessing
                ImageData{iImage}.img = normalizeimage(ImageData{iImage}.img ); 

                % convert image to array
                [ImageData{iImage}.imgVec, ImageData{iImage}.dims] = imgtoarray(ImageData{iImage}.img);

                % convert ground truth to array
                [ImageData{iImage}.gtVec, ImageData{iImage}.gtdims] = imgtoarray(ImageData{iImage}.gt);
            end
            
            % Get Data in appropriate form
            Options.trainPrct = {.3, .3};
            Options.labelPrct = {.1, .1};
            Data = getdataformat(ImageData, Options);
            clear ImageData;
            
            % Tune data
            ImageData = [];

            load('H:\Data\Images\RS\lidar_hsi\Main\image1.mat');
            ImageData{1}.img = Image1(:,:,1:247);             % 400m image
            clear Image1;

            load('H:\Data\Images\RS\lidar_hsi\Main\image3.mat');
            ImageData{2}.img = Image3;             % 2000m image
            clear Image2;

            load('H:\Data\Images\RS\lidar_hsi\Main\imggt.mat');
            ImageData{1}.gt = imgGT;      % 400m image ground truth
            ImageData{2}.gt = imgGT;      % 2000m image ground truth
            clear imgGT;
            
            
        end
            
          
        
        
    otherwise
        error('Unrecognized dataset chosen.');
        
end





%=================================\
%% Manifold Alignment
%=================================%

% Semisupervised Manifold Alignment
Options = [];

% adjacency matrix options
AdjacencyOptions = [];
AdajcencyOptions.type = 'standard';
AdjacencyOptions.nn_graph = 'knn';
AdjacencyOptions.k = 20;
AdjacencyOptions.kernel = 'heat';
AdjacencyOptions.sigma = 1;
AdjacencyOptions.saved = 0;

Options.AdjacencyOptions = AdjacencyOptions;

% spatial spectral potential matrix options
PotentialOptions = [];
PotentialOptions.type = 'spaspec';
PotentialOptions.clusterSigma = 1;
PotentialOptions.clusterkernel = 'heat';
PotentialOptions.weightSigma = 1;

% spatial adjacency matrix options
SpatialAdjacency.type = 'standard';
SpatialAdjacency.nn_graph = 'knn';
SpatialAdjacency.k = 4;
SpatialAdjacency.kernel = 'heat';
SpatialAdjacency.sigma = 1;
SpatialAdjacency.saved = 0;

% save spatial adjacency matrix options
PotentialOptions.SpatialAdjacency = SpatialAdjacency;

% save potential options
Options.PotentialOptions = PotentialOptions;


%% SSMA
% manifold alignment options
AlignmentOptions = [];
AlignmentOptions.type = 'ssma';
AlignmentOptions.nComponents = 'default';
AlignmentOptions.printing = 0;
AlignmentOptions.lambda = 0;
AlignmentOptions.mu = .2;
AlignmentOptions.alpha = .2;

% save Alignment options
Options.AlignmentOptions = AlignmentOptions;

tic;
% Manifold Alignment
projectionFunctions = manifoldalignment(Data, Options);

% Projection Functions

embedding.ssma = manifoldalignmentprojections(Data, projectionFunctions, 'ssma');

% Classification
ClassOptions = [];
ClassOptions.method = 'lda';
ClassOptions.dimStep = 4;
ClassOptions.nComponents = 10;
ClassOptions.exp = 'best';

stats.ssma = alignmentclassification(Data, embedding.ssma, ClassOptions);
stats.ssma{1,1}.time = toc;

%% Wang
% manifold alignment options
AlignmentOptions = [];
AlignmentOptions.type = 'wang';
AlignmentOptions.nComponents = 'default';
AlignmentOptions.printing = 0;
AlignmentOptions.lambda = 0;
AlignmentOptions.mu = .2;
AlignmentOptions.alpha = .2;

% save Alignment options
Options.AlignmentOptions = AlignmentOptions;

% Manifold Alignment
tic;
projectionFunctions = manifoldalignment(Data, Options);

% Projection Functions

embedding.wang = manifoldalignmentprojections(Data, projectionFunctions, 'wang');


% Classification
ClassOptions = [];
ClassOptions.method = 'lda';
ClassOptions.dimStep = 4;
ClassOptions.nComponents = 10;
ClassOptions.exp = 'best';

stats.wang = alignmentclassification(Data, embedding.wang, ClassOptions);
stats.wang{1,1}.time = toc;

%% SEMA
% manifold alignment options
AlignmentOptions = [];
AlignmentOptions.type = 'sema';
AlignmentOptions.nComponents = 'default';
AlignmentOptions.printing = 0;
AlignmentOptions.lambda = 0;
AlignmentOptions.mu = .2;
AlignmentOptions.alpha = .2;

% save Alignment options
Options.AlignmentOptions = AlignmentOptions;

% Manifold Alignment
tic;
projectionFunctions = manifoldalignment(Data, Options);

% Projection Functions

embedding.sema = manifoldalignmentprojections(Data, projectionFunctions, 'sema');


% Classification
ClassOptions = [];
ClassOptions.method = 'lda';
ClassOptions.dimStep = 4;
ClassOptions.nComponents = 10;
ClassOptions.exp = 'best';

stats.sema = alignmentclassification(Data, embedding.sema, ClassOptions);
stats.sema{1,1}.time = toc;

%% Plots
TableOptions = [];
TableOptions.exp = 'alignment';


classNames = {'1', '2', '3', '4', '5', '6', '7'};
stats.classNames = classNames;
bestclassresults(stats, TableOptions)


    