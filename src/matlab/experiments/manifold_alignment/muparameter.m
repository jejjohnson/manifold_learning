% Mu-Alpha Parameter Estimation


% Clear Variables
clear all; close all; clc;

% dataset
dataset = 'vcu';

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
        
    otherwise
        error('Unrecognized dataset chosen.');
end



%=================================\
%% Manifold Alignment Parameters
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

% Manifold Alignment Options
AlignmentOptions = [];

AlignmentOptions.nComponents = 'default';
AlignmentOptions.printing = 0;
AlignmentOptions.lambda = 0;

%===============
% Experiment Options
%==================================%

muParameters = logspace(-1,2,10);
alphaParameters = logspace(-1,2,10);
cases = 1:4;
trainPrctParams = [.25, .8];

trainCount = 1;

embedding = [];
rng('default');         % reproducibility


g = waitbar(0, 'Initializing waitbar...');      % training prct waitbar


for iTrainPrct = trainPrctParams
    
    % choose training percentage
    Options.trainPrct = iTrainPrct;
    
    
    waitbar(trainCount/numel(trainPrctParams), g, ...
        sprintf('Training Percent: %.2f', iTrainPrct));
    
    f = waitbar(0, 'Initializing waitbar...');  % cases waitbar
    
    caseCount = 1;
    for icase = cases
        
        % choose case
        
        waitbar(caseCount/numel(cases), f, ...
        sprintf('Case Number: %d', icase));
        
        switch icase
            
            case 1
                Options.labelPrct = {.1, .1};
                
            case 2
                Options.labelPrct = {.1, .5};
            
            case 3
                Options.labelPrct = {.5, .1};
                
            case 4
                Options.labelPrct = {.5, .5};
                
            otherwise
                error('Unrecognized case.');
        end
        
        h = waitbar(0, 'Initializing waitbar...');
        
        muCount = 1;
        for iMu = muParameters


            % Get Data in appropriate form
            Options.trainPrct = {.1, .1};
            Options.labelPrct = {.1, .1};
            Data = getdataformat(ImageData, Options);

            %-----------------------------------%
            % SemiSupervised Manifold Alignment %
            %-----------------------------------%

            waitbar(muCount/numel(muParameters), h, ...
                sprintf('SSMA - \\mu: %.4f', iMu));

            % manifold alignment options
            AlignmentOptions.mu = iMu;
            AlignmentOptions.type = 'ssma';    

            % save Alignment options
            Options.AlignmentOptions = AlignmentOptions;

            % Manifold Alignment
            projectionFunctions = manifoldalignment(Data, Options);

            % Embedding
            embedding.ssma = manifoldalignmentprojections(Data, projectionFunctions, 'ssma');
            
            % statistics
            ClassOptions = [];
            ClassOptions.dimStep = 4;
            
            ClassOptions.method = 'lda';
            stats.ssmalda{muCount} = alignmentclassification(Data, embedding.ssma, ClassOptions);
            
            ClassOptions.method = 'svm';
            stats.ssmasvm{muCount} = alignmentclassification(Data, embedding.ssma, ClassOptions);


            %-----------------------------------%
            % Wang Original Method  %
            %-----------------------------------%

            waitbar(muCount/numel(muParameters), h, ...
                sprintf('Wang - \\mu: %.4f', iMu));
            % manifold alignment options
            AlignmentOptions.mu = iMu;
            AlignmentOptions.type = 'wang';    

            % save Alignment options
            Options.AlignmentOptions = AlignmentOptions;

            % Manifold Alignment
            projectionFunctions = manifoldalignment(Data, Options);

            % Embedding
            embedding.wang = manifoldalignmentprojections(Data, projectionFunctions, 'ssma');
            
            % statistics
            ClassOptions = [];
            ClassOptions.dimStep = 4;

            ClassOptions.method = 'lda';
            stats.wanglda{muCount} = alignmentclassification(Data, embedding.wang, ClassOptions);
            
            ClassOptions.method = 'svm';
            stats.wangsvm{muCount} = alignmentclassification(Data, embedding.wang, ClassOptions);


            %--------------------------------------%
            % Spatial-Spectral Schroedinger Method %
            %--------------------------------------%

            alphaCount = 1;     % index counter

            for iAlpha = alphaParameters

                waitbar(muCount/numel(muParameters), h, ...
                sprintf('SSSEMA - \\mu: %.4f, \\alpha: %.4f', iMu, iAlpha));

                % manifold alignment options
                AlignmentOptions.mu = iMu;
                AlignmentOptions.type = 'sssema'; 
                AlignmentOptions.alpha = .2;

                % save Alignment options
                Options.AlignmentOptions = AlignmentOptions;

                % Manifold Alignment
                projectionFunctions = manifoldalignment(Data, Options);

                % Embedding
                embedding.sema = ...
                    manifoldalignmentprojections(Data, projectionFunctions, 'ssse');
                
                % statistics
                ClassOptions = [];
                ClassOptions.dimStep = 4;

                ClassOptions.method = 'lda';
                stats.semalda{muCount, alphaCount} = alignmentclassification(Data, embedding.sema, ClassOptions);

                ClassOptions.method = 'svm';
                stats.semasvm{muCount, alphaCount} = alignmentclassification(Data, embedding.sema, ClassOptions);

                % alpha parameter counter
                alphaCount = alphaCount + 1;
            end

            % mu parameter counter
            muCount = muCount + 1;
            


        end

        close(h);
        %%%%%%%%%%%%%%
        %% Save Data 
        %%%%%%%%%%%%%%
        
        save_str = 'H:\Data\saved_data\manifold_alignment\parameter_estimation\';
        save_str = [save_str sprintf('%s_train%d_case%d', dataset, trainCount, icase)];
        save(save_str, 'stats', 'Options');
        
        
        caseCount = caseCount + 1;
    end
    
    close(f);
    trainCount = trainCount + 1;
end

close(g);
    
    
