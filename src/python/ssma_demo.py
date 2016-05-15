from data.data_generation import generate_gaussian
from manifold_alignment.ssma import ManifoldAlignment
import numpy as np
from utils.classification_list import lda_pred, accuracy_stats


# my test function to see if it works. Should be in the 80s range of
# accuracy
def test_ma_gaussian(ma_method='wang', n_components=2, plot=False):



    # define some dictionaries with empty labeled lists
    X ={}; Y={};
    X['label'] = []; X['unlabel'] = []; X['test'] = []
    Y['label'] = []; Y['unlabel'] = []; Y['test'] = []


    # assign labels from gaussian dataset
    X1, X2, XT1, XT2, \
    Y1, Y2, YT1, YT2, \
    U1, U2 = generate_gaussian(plot_data=plot)


    # create appropriate data structures based off of
    # the manifold alignment class criteria
    X['label'] = [X1, X2]
    X['unlabel'] = [U1, U2]
    X['test'] = [XT1, XT2]
    Y['label'] = [Y1 , Y2]
    Y['test'] = [YT1, YT2]

    print np.shape(X['label'][0]), np.shape(Y['label'][0])
    print np.shape(X['unlabel'][0])
    print np.shape(X['test'][0]), np.shape(Y['test'][0])

    print np.shape(X['label'][1]), np.shape(Y['label'][1])
    print np.shape(X['unlabel'][1])
    print np.shape(X['test'][1]), np.shape(Y['test'][1])

    ma_method = ManifoldAlignment(ma_method=ma_method,
                                  lap_method='personal')
    ma_method.fit(X,Y)
    Xproj = ma_method.transform(X, n_components=2)


    Y['pred'] = lda_pred(Xproj['train'],
                     Xproj['test'],
                     Y['label'],
                     Y['test'])

    Acc_stats = accuracy_stats(Y['pred'], Y['test'])


    Lg = ma_method.L_g
    Vs = ma_method.V_s
    Vd = ma_method.V_d

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows=3, ncols=1,
                           figsize=(10,10))

    ax[0].spy(Lg, precision=1E-5, markersize=.2)
    ax[0].set_title('Geometric Laplacian')
    ax[1].spy(Vs, precision=1E-5, markersize=.2)
    ax[1].set_title('Similarity Potential')
    ax[2].spy(Vd, precision=1E-5, markersize=.2)
    ax[2].set_title('Dissimilarity Potential')

    plt.show()

    print('AA - Domain 1: {s}'.format(s=Acc_stats['AA'][0]))
    print('AA - Domain 2: {s}'.format(s=Acc_stats['AA'][1]))


if __name__ == "__main__":

    test_ma_gaussian(ma_method='ssma', n_components=3, plot=True)
