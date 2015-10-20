# -*- coding: utf-8 -*-

"""K-mode clustering"""

import warnings
from scipy.sparse import coo_matrix
import math
from math import log
import numpy as np
import scipy.sparse as sp
from scipy.misc import comb

import _k_modes

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score, \
    normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state, as_float_array, check_random_state
from sklearn.utils.extmath import row_norms, squared_norm, safe_sparse_dot
from sklearn.metrics import silhouette_score

from sklearn.cluster import _k_means


def _compute_all_irm(X, n_clusters):
    matrix_all_irm = np.zeros((X.shape[0], X.shape[0]))
    matrix_all_irm.fill(1)

    for i in range(X.shape[0]):
        for j in range(i+1, X.shape[0]):
            irm = interdep_redund_measure(X[i],X[j])
            matrix_all_irm[i][j] = irm
            matrix_all_irm[j][i] = irm

    return matrix_all_irm

def k_modes(X, n_clusters, n_init=1, max_iter=5,
            verbose=False, tol=1e-4, random_state=None, copy_x=True, n_jobs=1):
    """K-modes clustering algorithm."""
    if n_init <= 0:
        raise ValueError("Invalid number of initializations."
                         " n_init=%d must be bigger than zero." % n_init)

    X = as_float_array(X, copy=copy_x)
    matrix_all_irm = _compute_all_irm(X, n_clusters)
    best_labels, best_modes, best_mirm = None, None, -np.inf

    if n_jobs == 1:

        for j in range(2,n_clusters+1):
            # For a single thread, less memory is needed if we just store one set
            # of the best results (as opposed to one set per run per thread).
            for it in range(n_init):
                # run a k-modes once
                labels, modes, mirm_sum = _kmodes_single(
                    X, j, matrix_all_irm, max_iter=max_iter,
                    verbose=verbose, tol=tol, random_state=random_state)
                # determine if these results are the best so far
                if mirm_sum >= best_mirm:
                    best_labels = labels.copy()
                    best_modes = modes.copy()
                    best_mirm = mirm_sum
    else:
        # TODO:
        pass

    return best_modes, best_labels, best_mirm

def _kmodes_single(X, k, matrix_all_irm, max_iter=5,
                   verbose=False, random_state=None, tol=1e-4):
    """A single run of k-modes, assumes preparation completed prior."""
    best_labels, best_modes = None, None
    # init
    modes, modes_labels = _init_modes(X, k, random_state=random_state)
    modes_labels.sort()
    # X = X.T
    if verbose:
        print("Initialization complete")

    # iterations
    for i in range(max_iter):
        # modes_old = modes.copy()
        modes_labels_old = modes_labels.copy()
        # labels_old = _labels.copy()
        # labels assignment
        labels = _labels(X, modes_labels, matrix_all_irm)

        #compute new modes of groups
        modes_labels, all_multiple_irm = _recompute_modes(X, labels, modes_labels, modes, matrix_all_irm)
        if verbose:
            print("Iteration %2d" % i)

        if i >= max_iter or (modes_labels_old == modes_labels).all():
            if verbose:
                print("Converged at iteration %d" % i)
            break

    modes_labels.sort()
    return labels, modes_labels, all_multiple_irm.sum()

def _recompute_modes(X, labels, modes_label, modes, matrix_all_irm):
    """Recompute modes of groups.
    X: data
    labels: labels of atributes
    modes: modes of groups
    """
    aux = np.array(range(labels.size), dtype=float)
    multiple_irm = np.array(range(labels.size), dtype=float)

    aux.fill(-np.inf)

    for i in range(modes_label.size):
        multiple_irm.fill(-np.inf)
        cont = 0
        max = -np.inf
        mask = (i == labels)
        modes_label_aux = np.array(range(labels.size))
        modes_label_aux = modes_label_aux[mask]

        # multiple_irm for each group
        multiple_irm[mask] = matrix_all_irm[modes_label_aux,:][:,modes_label_aux].sum(0)
        aux[mask] = multiple_irm[mask]
        modes_label[i] = multiple_irm.argmax()

    multiple_irm = aux
    return modes_label, multiple_irm

def _labels(X, seeds, matrix_all_irm):
    """Compute the labels of the given samples and modes.

    Returns
    -------
    labels: int array of shape(n)
        The resulting assignment
    """

    labels = np.empty(X.shape[0], dtype=np.int32)
    labels.fill(-np.inf)
    irm_labels = matrix_all_irm[seeds]
    max_irm = np.max(irm_labels,0)

    for i in range(seeds.size):
        labels[max_irm == irm_labels[i]] = i

    return labels

def _init_modes(X, k, random_state=None, x_squared_norms=None,
                    init_size=None):
    """Compute the initial centroids

    Returns
    -------
    modes: array, shape(k, n_features)
    """
    random_state = check_random_state(random_state)
    n_samples = X.shape[0]

    if init_size is not None and init_size < n_samples:
        if init_size < k:
            warnings.warn(
                "init_size=%d should be larger than k=%d. "
                "Setting it to 3*k" % (init_size, k),
                RuntimeWarning, stacklevel=2)
            init_size = 3 * k
        init_indices = random_state.random_integers(
            0, n_samples - 1, init_size)
        X = X[init_indices]
        x_squared_norms = x_squared_norms[init_indices]
        n_samples = X.shape[0]
    elif n_samples < k:
        raise ValueError(
            "n_samples=%d should be larger than k=%d" % (n_samples, k))

    seeds = random_state.permutation(n_samples)[:k]
    modes = X[seeds]

    if sp.issparse(modes):
        modes = modes.toarray()

    if len(modes) != k:
        raise ValueError('The shape of the initial modes (%s) '
                         'does not match the number of clusters %i'
                         % (modes.shape, k))

    return modes, seeds

def interdep_redund_measure(labels_true, labels_pred):

    contingency = contingency_matrix(labels_true, labels_pred)
    contingency = np.array(contingency, dtype='float')
    contingency_sum = np.sum(contingency)
    pi = np.sum(contingency, axis=1)
    pj = np.sum(contingency, axis=0)
    outer = np.outer(pi, pj)
    nnz = contingency != 0.0
    # normalized contingency
    contingency_nm = contingency[nnz]
    log_contingency_nm = np.log(contingency_nm)
    contingency_nm /= contingency_sum
    # log(a / b) should be calculated as log(a) - log(b) for
    # possible loss of precision
    log_outer = -np.log(outer[nnz]) + log(pi.sum()) + log(pj.sum())

    joint_entropy = np.sum(contingency_nm * log_contingency_nm + contingency_nm * log_outer)

    mi = np.sum(contingency_nm * (log_contingency_nm - log(contingency_sum))
          + contingency_nm * log_outer)

    return mi/joint_entropy


def contingency_matrix(labels_true, labels_pred, eps=None):
    """Build a contengency matrix describing the relationship between labels.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference

    labels_pred : array, shape = [n_samples]
        Cluster labels to evaluate

    eps: None or float
        If a float, that value is added to all values in the contingency
        matrix. This helps to stop NaN propagation.
        If ``None``, nothing is adjusted.

    Returns
    -------
    contingency: array, shape=[n_classes_true, n_classes_pred]
        Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
        true class :math:`i` and in predicted class :math:`j`. If
        ``eps is None``, the dtype of this array will be integer. If ``eps`` is
        given, the dtype will be float.
    """
    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    # Using coo_matrix to accelerate simple histogram calculation,
    # i.e. bins are consecutive integers
    # Currently, coo_matrix is faster than histogram2d for simple cases
    contingency = coo_matrix((np.ones(class_idx.shape[0]),
                              (class_idx, cluster_idx)),
                             shape=(n_classes, n_clusters),
                             dtype=np.int).toarray()
    if eps is not None:
        # don't use += as contingency is integer
        contingency = contingency + eps
    return contingency

class KModes():
    """K-Modes clustering
    Parameters
    ----------"""

    def __init__(self, n_clusters=8, n_init=5, max_iter=5,
                 tol=1e-4, verbose=0, random_state=None, copy_x=True, n_jobs=1):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Compute k-modes clustering.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)"""
        random_state = check_random_state(self.random_state)
        X = np.round(X.T,1)

        self.cluster_modes_, self.labels_, self.n_mirm_= \
            k_modes(
                X, n_clusters=self.n_clusters, n_init=self.n_init,
                max_iter=self.max_iter, verbose=self.verbose, tol=self.tol,
                random_state=random_state, copy_x=self.copy_x, n_jobs=self.n_jobs)
        return self
