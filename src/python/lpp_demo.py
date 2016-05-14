import matplotlib.pyplot as plt
plt.style.use('ggplot')

from time import time

from sklearn import manifold, datasets
from sklearn.manifold import SpectralEmbedding
from lpproj import LocalityPreservingProjection
from manifold_learning.lpp import LocalityPreservingProjections

def swiss_roll_test():



    n_points = 1000
    X, color = datasets.samples_generator.make_s_curve(n_points,
                                                       random_state=0)
    n_neighbors=20
    n_components=2

    # original lE algorithm


    t0 = time()
    ml_model = SpectralEmbedding(n_neighbors=n_neighbors,
                                 n_components=n_components)
    Y = ml_model.fit_transform(X)
    t1 = time()

    # 2d projection
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(5,10))
    ax[0].scatter(Y[:,0], Y[:,1], c=color, label='scikit')
    ax[0].set_title('Sklearn-LE: {t:.2g}'.format(t=t1-t0))


    # Jakes LPP Algorithm

    t0 = time()
    ml_model = LocalityPreservingProjection(n_components=n_components)
    ml_model.fit(X)
    Y = ml_model.transform(X)
    t1 = time()

    ax[1].scatter(Y[:,0], Y[:,1], c=color, label='Jakes Algorithm')
    ax[1].set_title('Jakes LPP: {t:.2g}'.format(t=t1-t0))

    # my SSSE algorith,

    t0 = time()
    ml_model = LocalityPreservingProjections(weight='angle',
                                             n_components=n_components,
                                             n_neighbors=n_neighbors,
                                             sparse=True,
                                             eig_solver='dense')
    ml_model.fit(X)
    Y = ml_model.transform(X)
    t1 = time()

    ax[2].scatter(Y[:,0], Y[:,1], c=color, label='My LPP Algorithm')
    ax[2].set_title('My LPP: {t:.2g}'.format(t=t1-t0))

    plt.show()

if __name__ == "__main__":
    swiss_roll_test()
