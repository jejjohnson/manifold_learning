% Run Experiments
clear all; close all; clc;







% % run alpha parameter experiment (Indian Pines)
% tic;
% display('Indian Pines Dimension Exp...');
% run dimsindianpines.m
% display('Done!');
% toc;

% run # of samples experiment (embedding)
tic;
display('Pavia Dimension Exp...');
run dimspavia.m
display('Done!');
toc;

%% Schroedinger Manifold Alignment Experiments

% Run ultimate experiment VCU Data (.2, .8)
display('Running Manifold Alignment Experiment on VCU data...');
run muparameter.m
display('Done!');
