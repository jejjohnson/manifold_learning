%{
This is a settings file which has all of the necessary MATLAB structs
needed to run a complete experiment.
%}

%-- KERNEL EIGENMAP METHODS --%
%{
Kernel Eigenmap Methods available
* Laplacian Eigenmaps (LE) (default)
* Spatial-Spectral Schroedinger Eigenmaps (SpaSpeSE)
* Semi-Supervised Schroedinger Eigenmaps (SSSE)
* Spatial-Spectral Semi-Supervised Schroedinger Eigenmaps (SpaSpeSSSE)
%}
KESettings.method = 'LE';

%-- ADJACENCY MATRIX --%
AdjacencySettings.adjacencyType = 'nn';
AdjacencySettings.matrixType = 'sparse';     % ['sparse', 'dense'] (str)

%-- K-NEAREST NEIGHBORS --%
knnSettings.type = 'standard';
knnSettings.nn_graph = 'knn';       % ['knn', 'kdtree', 'flann', 'radius']
knnSettings.k = 20;                 % num of nearest neighbors (int)
knnSettings.r = 20;                 % radius for epsilon ball (int)

%-- WEIGHTING FUNCTION --%
knnSettings.kernel = 'heat';        % Options 'cosine', 'heat', 'simple'
knnSettings.sigma = 1.0;

%-- TIME SAVING MEASURES --%
knnSettings.savedfile = 0;          % no saved file location

%-- LAPLACIAN MATRIX --%
%{ 
Laplacian Matrix Types:
* Normalized
* Unnormalized (default)
* Random Walk
* Renormalized
* Geometric
%}
LaplacianSettings.type = 'unnormalized';

%-- CONSTRAINT MATRIX --%
%{
Constaint used in Ax = lambda Bx, where the constraint is B. 
Constraint Matrix Types:
* Identity (ratio cut)
* Degree (normalized cuts) (default)
* Dissimilarity
* K-Scaling (TODO)
* Identity + beta * Dissimilarity (TODO)
* Degree + beta * Dissimilarity (TODO)
%}
ConstraintSettings.type = 'degree';
ConstraintSettings.beta = 0.1;


%-- DIMENSION REDUCTION --%
eigSettings.numEigs = 10;

%% Method Specific Settings

%-- Spatial-Spectral Schroedinger Eigenmaps --%

% spatial-spectral type
SpaSpeSettings.ssType = 'spaspec'; % ['spaspe', 'spespa']

% Spectral NN Settings
SpeKnnSettings.adjacencyType = 'nn';
SpeKnnSettings.matrixType = 'sparse';
SpeKnnSettings.type = 'standard';
SpeKnnSettings.nnGraph = 'knn';
SpeKnnSettings.k = 20;
SpeKnnSettings.r = 20;
SpeKnnSettings.kernel = 'heat';
SpeKnnSettings.sigma = 1.0;

% spatial knn
SpaKnnSettings.adjacencyType = 'nn';
SpaKnnSettings.matrixType = 'sparse';
SpaKnnSettings.type = 'standard';
SpaKnnSettings.nnGraph = 'knn';
SpaKnnSettings.k = 4;
SpaKnnSettings.r = 20;
SpaKnnSettings.kernel = 'heat';
SpaKnnSettings.sigma = 1.0;

% 



%-- Semi-Supervised Schroedinger Eigenmaps --%








