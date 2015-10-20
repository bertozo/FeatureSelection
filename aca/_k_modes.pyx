# # -*- coding: utf-8 -*-
#
# from sklearn.metrics import mutual_info_score
# import numpy as np
# cimport numpy as np
#
# ctypedef np.float64_t DOUBLE
# ctypedef np.int32_t INT
#
#
# def _x_compute_all_irm(np.ndarray[DOUBLE, ndim=2] X, int n_clusters):
#
#     cdef:
#         np.ndarray[DOUBLE, ndim=2] matrix_all_irm = np.zeros((X.shape[0], X.shape[0]))
#         double irm
#         unsigned int i, j
#
#     matrix_all_irm.fill(1)
#
#     for i in range(X.shape[0]):
#         print i," de: ", X.shape[0]
#         for j in range(i+1, X.shape[0]):
#             try:
#                 irm = interdep_redund_measure(X[i],X[j])
#             except:
#                 icmi = 0
#             matrix_all_irm[i][j] = irm
#             matrix_all_irm[j][i] = irm
#
#     return matrix_all_irm
#
#
# cdef double interdep_redund_measure(np.ndarray[DOUBLE, ndim=1] X,
#                                     np.ndarray[DOUBLE, ndim=1] Y):
#     # return mutual_info_score(None, None,contingency=(X,Y))
#     return mutual_info_score(X, Y)