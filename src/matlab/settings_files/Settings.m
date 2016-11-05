%{
This is a settings file which has all of the necessary MATLAB structs
needed to run a complete experiment.
%}

%-- ADJACENCY MATRIX --%


%-- K-NEAREST NEIGHBORS --%
knnSettings.type = 'standard';
knnSettings.nn_graph = 'knn';       % ['knn', 'kdtree', 'flann', 'radius']
knnSettings.k = 20;                 % int
knnSettings.kernel = 'heat';        % Options 'cosine', 'heat', 'simple'
knnSettings.sigma = 1.0;
knnSettings.savedfile = 0;          % no saved file location


%-- DIMENSION REDUCTION --%

eigSettings.numEigs = 10;
% Constaint used in Ax = lambda Bx, where the constraint is B. 
% Options include:
% * Identity
% * Degree (DEFAULT)
% * k-Scaling (TODO)
% * dissimilarity (TODO)
% * Combination
eigSettings.constraint = 'degree';  




