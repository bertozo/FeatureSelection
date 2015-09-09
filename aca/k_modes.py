# -*- coding: utf-8 -*-

"""K-mode clustering"""

import warnings

import math
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
from sklearn.utils.sparsefuncs import mean_variance_axis
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

def interdep_redund_measure(X, Y):
    return mutual_info_score(None, None,contingency=(X,Y))
    # return mutual_info_score(X,Y)

def k_modes(X, n_clusters, n_init=10, max_iter=300,
            verbose=False, tol=1e-4, random_state=None, copy_x=True, n_jobs=1,
            return_n_iter=False):
    """K-modes clustering algorithm."""
    if n_init <= 0:
        raise ValueError("Invalid number of initializations."
                         " n_init=%d must be bigger than zero." % n_init)

    X = as_float_array(X, copy=copy_x)
    # matrix_all_irm = _compute_all_irm(X, n_clusters)
    matrix_all_irm = _k_modes._x_compute_all_irm(X, n_clusters)
    best_labels, best_modes = None, None
    ss_old = -np.inf
    if n_jobs == 1:
        # For a single thread, less memory is needed if we just store one set
        # of the best results (as opposed to one set per run per thread).
        for it in range(n_init):
            # run a k-modes once
            labels, modes, n_iter_, ss= _kmodes_single(
                X, n_clusters, matrix_all_irm, max_iter=max_iter,
                verbose=verbose, tol=tol, random_state=random_state)
            # determine if these results are the best so far
            if ss > ss_old:
                best_labels = labels.copy()
                best_modes = modes.copy()
                iter = n_iter_
                ss_old = ss
    else:
        # TODO:
        pass

    if return_n_iter:
        return best_modes, best_labels, iter
    else:
        return best_modes, best_labels

def _kmodes_single(X, n_clusters, matrix_all_irm, max_iter=300,
                   verbose=False, random_state=None, tol=1e-4):
    """A single run of k-modes, assumes preparation completed prior."""
    best_labels, best_modes = None, None
    # init
    modes, seeds = _init_modes(X, n_clusters, random_state=random_state)
    # X = X.T
    if verbose:
        print("Initialization complete")

    # iterations
    for i in range(max_iter):
        modes_old = modes.copy()
        # labels assignment
        labels = _labels(X, modes, seeds, matrix_all_irm, n_clusters)

        #compute new modes of groups
        modes, seeds = _recompute_modes(X, labels, seeds, modes, n_clusters, matrix_all_irm)

        if verbose:
            print("Iteration %2d" % i)

        if (modes_old == modes).all() or i >= max_iter:
            if verbose:
                print("Converged at iteration %d" % i)
            break

    ss = silhouette_score(1 - matrix_all_irm,labels,metric='precomputed')
    return labels, modes, i + 1, ss

def _recompute_modes(X, labels, seeds, modes, n_clusters, matrix_all_irm):
    """Recompute modes of groups.
    X: data
    labels: labels of atributes
    modes: modes of groups
    n_clusters: num of modes
    """
    seeds_aux = np.array(range(labels.size))
    aux = np.array([])

    for i in range(n_clusters):
        cont = 0
        max = -np.inf
        mask = i == labels
        seeds_aux = np.array(range(len(labels)))
        seeds_aux = seeds_aux[mask]

        for j in matrix_all_irm[seeds_aux]:
            sum_r = np.sum(j[mask])
            if sum_r > max:
                max = sum_r
                seeds[i] = seeds_aux[cont]

            cont += 1

        modes[i] = X[seeds[i]]

    return modes, seeds

def _labels(X, modes, seeds, matrix_all_irm, n_clusters):
    """Compute the labels of the given samples and modes.

    Returns
    -------
    labels: int array of shape(n)
        The resulting assignment
    """

    labels = np.empty(X.shape[0], dtype=np.int32)
    labels.fill(-1)
    irm_labels = matrix_all_irm[seeds]
    max_irm = np.max(irm_labels,0)

    for i in range(n_clusters):
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


class KModes():
    """K-Modes clustering
    Parameters
    ----------"""

    def __init__(self, n_clusters=8, n_init=10, max_iter=300,
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
        X = X.T

        self.cluster_centers_, self.labels_, self.n_iter_ = \
            k_modes(
                X, n_clusters=self.n_clusters, n_init=self.n_init,
                max_iter=self.max_iter, verbose=self.verbose,
                return_n_iter=True, tol=self.tol, random_state=random_state,
                copy_x=self.copy_x, n_jobs=self.n_jobs)
        return self
