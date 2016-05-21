import matplotlib.pyplot as plt
from time import time

from mpl_toolkits.mplot3d import Axes3D

from sklearn import manifold, datasets
from sklearn import manifold

from manifold_learning.se import SchroedingerEigenmaps

# swiss roll test to test out my function versus theirs
def swiss_roll_test():


    n_points = 1000
    X, color = datasets.samples_generator.make_s_curve(n_points,
                                                       random_state=0)
    n_neighbors=20
    n_components=2

    # original lE algorithm


    t0 = time()
    ml_model = manifold.SpectralEmbedding(n_neighbors=n_neighbors,
                                          n_components=n_components)
    Y = ml_model.fit_transform(X)
    t1 = time()

    # 2d projection
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(5,10))
    ax[0].scatter(Y[:,0], Y[:,1], c=color, label='scikit')
    ax[0].set_title('Sklearn-LE: {t:.2g}'.format(t=t1-t0))
    # my LE algorith,

    t0 = time()
    ml_model = SchroedingerEigenmaps(affinity='heat',
                                     n_components=n_components,
                                     n_neighbors=n_neighbors,
                                     sparse=False,
                                     eig_solver='dense',)
    ml_model.fit(X)
    Y = ml_model.transform(X)
    t1 = time()

    ax[1].scatter(Y[:,0], Y[:,1], c=color, label='my algorithm')
    ax[1].set_title('LE: {t:.2g}'.format(t=t1-t0))

    # my SSSE algorith,

    t0 = time()
    ml_model = SchroedingerEigenmaps(affinity='heat',
                                     n_components=n_components,
                                     n_neighbors=n_neighbors,
                                     sparse=False,
                                     eig_solver='dense',
                                     potential='ssse')
    ml_model.fit(X)
    Y = ml_model.transform(X)
    t1 = time()

    ax[2].scatter(Y[:,0], Y[:,1], c=color, label='my algorithm')
    ax[2].set_title('SSSE: {t:.2g}'.format(t=t1-t0))

    plt.show()


if __name__ == "__main__":

     swiss_roll_test()
