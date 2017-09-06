import matplotlib.pyplot as plt
from time import time
from sklearn import (manifold, datasets)
from manifold_learning.se import SchroedingerEigenmaps
from data.get_hsi_data import get_data
from mpl_toolkits.mplot3d import Axes3D
Axes3D

# swiss roll test to test out my function versus theirs
def swiss_roll_test():

    n_points = 750
    X, color = datasets.samples_generator.make_s_curve(n_points,
                                                       random_state=0)
    fig = plt.figure(figsize=(5,10))
    ax = fig.add_subplot(311, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, label='Data', cmap=plt.cm.Spectral)
    ax.set_title('Original Dataset')

    n_neighbors=20
    n_components=2

    # Laplacian Eigenmaps (scikit-learn)
    t0 = time()
    ml_model = manifold.SpectralEmbedding(n_neighbors=n_neighbors,
                                          n_components=n_components)
    Y = ml_model.fit_transform(X)
    t1 = time()

    # 2D Projection
    ax = fig.add_subplot(312)
    ax.scatter(Y[:,0], Y[:,1], c=color, label='scikit', cmap=plt.cm.Spectral)
    ax.set_title('Sklearn-LE: {t:.2g}'.format(t=t1-t0))

    # Laplacian Eigenmaps (my version)
    t0 = time()
    ml_model = SchroedingerEigenmaps(n_components=n_components,
                                     n_neighbors=n_neighbors)
    Y = ml_model.fit_transform(X)
    t1 = time()

    # 2D Projection
    ax = fig.add_subplot(313)
    ax.scatter(Y[:,0], Y[:,1], c=color, label='my algorithm', cmap=plt.cm.Spectral)
    ax.set_title('LE: {t:.2g}'.format(t=t1-t0))

    # Todo: Schroedinger Eigenmaps (Spatial Spectral Potential)
    # Todo: Schroedinger Eigenmaps (Partial Labels Potential)

    plt.show()

def hsi_test():

    # get indian pines data
    img = get_data()



if __name__ == "__main__":

     swiss_roll_test()
     #hsi_test()
