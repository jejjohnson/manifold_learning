% Run Experiments
clear all; close all; clc;


%% Schroedinger Manifold Alignment Experiments

% Run ultimate experiment VCU Data (.2, .8)
display('Running Manifold Alignment Experiment on VCU data...');
run muparameter.m
display('Done!');


%% Schroedinger Eigenmap Projection Experiments



% run alpha parameter experiment (Indian Pines)
tic;
display('Running Alpha Parameter Experiment...');
run alphaparameter.m
display('Done!');
toc;


% run # of samples experiment (embedding)
tic;
display('Running embedded number of samples experiment...');
run samplesexperiment.m
display('Done!');
toc;

% run # of samples experiment (projection)
tic;
display('Running projection number of samples experiment...');
run projectionsamples.m
display('Done!');
toc;


