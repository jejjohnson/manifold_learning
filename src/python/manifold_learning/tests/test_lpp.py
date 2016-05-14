
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
from sklearn.datasets import make_blobs

from .. import LocalityPreservingProjections


def test_estimator_checks():
    """Run scikit-learn's suite of basic estimator checks"""
    from sklearn.utils.estimator_checks import check_estimator
    check_estimator(LocalityPreservingProjection)
