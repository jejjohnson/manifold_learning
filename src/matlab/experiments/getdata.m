clear all; close all; clc;



% Projection Samples
switch lower(options.samplesprojdata)
    
    case 'indianpines'
        
        save_path = 'H:\Data\saved_data\projection_samples\';
        
        iSamples = options.samples;         % 10:10:80
        dataset = 'indianpines';
        algorithm = options.algorithm;      % le, se, lpp, sep
        load([save_path sprintf('%s_%s_sampl%d', dataset, algorithm, iSamples)]);
        
    case 'pavia'
        
        
        save_path = 'H:\Data\saved_data\projection_samples\';
        
        iSamples = options.samples;         % 10:10:80
        dataset = 'pavia';
        algorithm = options.algorithm;      % le, se, lpp, sep
        load([save_path sprintf('%s_%s_sampl%d', dataset, algorithm, iSamples)]);
        
    otherwise
        error('Invalid Adjacency matrix choice.');
end
        

% # of Samples datasets
switch lower(options.samplesdata)
    
    case 'indianpines'
        
        save_path = 'H:\Data\saved_data\samples_results\';
        
        iSamples = options.samples;         % 10:10:80
        dataset = 'indianpines';
        algorithm = options.algorithm;      % le, se, lpp, sep
        load([save_path sprintf('%s_%s_%d', dataset, algorithm, iSamples)]);
        
    case 'pavia'
        
        
        save_path = 'H:\Data\saved_data\samples_results\';
        
        iSamples = options.samples;         % 10:10:80
        dataset = 'pavia';
        algorithm = options.algorithm;      % le, se, lpp, sep
        load([save_path sprintf('%s_%s_%d', dataset, algorithm, iSamples)]);
        
    otherwise
        error('Invalid Adjacency matrix choice.');
end



% K-NN Parameter Matrices

switch lower(options.knndataset)
    
    case 'indianpines'
        
        save_path = 'H:\Data\saved_data\adjacency\';
        
        ik = options.knn;
        dataset = 'IndianPines';
        load([save_path sprintf('%s_k%d', dataset, ik)]);
        
    case 'pavia'
        
        
        save_path = 'H:\Data\saved_data\adjacency\';
        
        ik = options.knn;
        dataset = 'Pavia';
        load([save_path sprintf('%s_k%d', dataset, ik)]); 
    otherwise
        error('Invalid Adjacency matrix choice.');
end


% Sigma Parameter Matrices
switch lower(options.sigmadataset)
    
    case 'indianpines'
        
        save_path = 'H:\Data\saved_data\sigma_results\';
        
        iSigma = options.sigma;
        dataset = 'IndianPines';
        load([save_path sprintf('sigma20_sigma%d', iSigma)]);
        
    case 'pavia'
        
        
        save_path = 'H:\Data\saved_data\sigma_results\';
        
        iSigma = options.sigma;
        dataset = 'Pavia';
        load([save_path sprintf('sigma20_sigma%d', iSigma)]);
        
    otherwise
        error('Invalid Adjacency matrix choice.');
end