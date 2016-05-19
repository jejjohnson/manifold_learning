import matplotlib.pyplot as plt
plt.style.use('ggplot')

from time import time
from sklearn import manifold, datasets
from sklearn.manifold import SpectralEmbedding
from manifold_learning.lpp import LocalityPreservingProjections
from manifold_learning.klpp import KernelLocalityPreservingProjections
from lpproj import LocalityPreservingProjection

def swiss_roll_test():



    n_points = 100
    X, color = datasets.samples_generator.make_s_curve(n_points,
                                                       random_state=0)

    n_neighbors=20
    n_components=2

    # Jakes LPP Algorithm
    t0 = time()
    ml_model = LocalityPreservingProjection(n_components=n_components)
    ml_model.fit(X)
    Y = ml_model.transform(X)
    t1 = time()

    # 2d projection
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(5,10))
    ax[0].scatter(Y[:,0], Y[:,1], c=color, label='jakes_lpp')
    ax[0].set_title('LPP - Jake: {t:.2g}'.format(t=t1-t0))



    # My LPP Algorithm
    t0 = time()
    ml_model = LocalityPreservingProjections(n_components=n_components)
    ml_model.fit(X)
    Y = ml_model.transform(X)
    t1 = time()

    ax[1].scatter(Y[:,0], Y[:,1], c=color, label='My Algorithm')
    ax[1].set_title('My LPP: {t:.2g}'.format(t=t1-t0))

    # my LPP algorithm

    t0 = time()
    ml_model = KernelLocalityPreservingProjections(weight='heat',
                                             n_components=n_components,
                                             n_neighbors=n_neighbors,
                                             sparse=False,
                                             eig_solver='sparse',
                                             kernel='rbf',
                                             gamma_kernel=1.2)
    ml_model.fit(X)
    Y = ml_model.transform(X)
    t1 = time()

    ax[2].scatter(Y[:,0], Y[:,1], c=color, label='My KLPP Algorithm')
    ax[2].set_title('My KLPP: {t:.2g}'.format(t=t1-t0))

    plt.show()

if __name__ == "__main__":
    swiss_roll_test()
