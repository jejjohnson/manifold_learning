## General Algorithm for Kernel Eigenmap Problems

1. Get Original Data
2. Set Appropriate Parameters (adjacency, laplacian, problem tuner)
3. Perform Manifold Learning
4. Get resulting data, correctly embedded

## List of Key Components 

1. Obtain Data
2. Transform Data (optional)
3. Create Adjacency Matrix
4. Create Laplacian Matrix
5. Create Constraint Matrix
6. Create Potential Matrix
7. Tune Eigenvalue Problem
8. Solve Eigenvalue Decomposition
9. Transform New Data (optional)

#### Adjacency Matrix

* Find the Nearest Neighbors (k-nn, simple, epsilon)
* Apply kernels to distances
* Construct matrix (sparse, dense)

#### Laplacian matrix

* Get Adjacency Matrix
* Get Diagonal Matrix
* Construct Laplacian Matrix (normalized, unormalized, random_walk, renormalized, geometric)

#### Constraint matrix

* Ratio Cut (identity)
* Normalized Cut (degree)
* Dissimilarity (dissimilarity)
* Combination of either (beta parameter for trade-off)

#### Eigenvalue Problem Tuner

* Set all matrices (Laplacian, Constraint, Semisupervised)
* Set all parameters (mu, alpha, beta)

#### Eigenvalue Decomposition

* set eigenvalue decomposition method
* tune the eigenvalue problem

---

# Classification Tools

* Training and Testing
* SVM Classification (w/ Parameter Tuner)
* LDA Classification
* Random Forest Classification (w/ Parameter Tuner)
* k-NN Classification (w/ Parameter Tuner)

---

# Statistics Tools

* Image Statistics
* Individual Classification Statistics
* Parameter Display
* ROC Curve
* AUC Curve

---

# Experiments

* Adjacency Matrix
* k-Neighbours, epsilon-radius
* Kernelization Scale
* Potential Matrix

---

# Dimension Reduction Algorithms

* Locality Preserving Projections (LPP)
* Laplacian Eigenmaps (LE)
* Schroedinger Eigenmaps (SE) (SSSE, SSSEPL)
* Schroedinger Eigenmap Projections (SEP) (SSSE, SSSEPL)

# Alignment Algorithms

* Manifold Alignment (MA) Wang et al.
* Semisupervised Manifold Alignment (SSMA) Tuia et al.
* Kernel Manifold Alignment (KEMA) Tuia et al.
* Schroedinger Manifold Alignment (SchrMA) Johnson et al.
* Manifold Alignment Regression (MAR) Johnson/Tuia/Camps-Valls

## Data Projections Alterations

* Random Projections
* Kernel Projections
* Orthogonal Projections
* Tensor Projections
* Regression Methods
* SuperPixels
* Nystroem (improved, enhanced, variational, locally linear landmarks)

## Speed Improvements
 
* k-NN Finders (FLANN, GPU, Parallel)
* Adjacency Matrix Construction(RF, entropic affinities)
* Eigenvalue Decompositions (RSVD, GPU, Multigrid, Regression)

