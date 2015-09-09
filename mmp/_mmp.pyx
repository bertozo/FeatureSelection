# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np
import time
import math

from scipy.stats import pearsonr
from operator import itemgetter

ctypedef np.int32_t INT
ctypedef np.float64_t DOUBLE

cdef inline float maximal_information_compresseion_index(np.ndarray Ai,
                                                         np.ndarray Aj):
    """
    :return: ICMI entre os atributos A e, Aj
    """
    cdef:
        float var_ai, var_aj, x, p, icmi

    var_ai = Ai.var()
    var_aj = Aj.var()
    x = var_ai + var_aj
    p = pearsonr(Ai, Aj)[0]
    icmi = x - math.sqrt(math.pow(x,2) - 4*var_ai*var_aj*(1 - math.pow(p,2)))
    if math.isnan(icmi):
        icmi = 1.0
    return round(icmi,5)

cpdef np.ndarray Compute_ICMI(np.ndarray data):

    cdef:
        unsigned int i, j
        int data_shape
        float icmi
        np.ndarray data_transpose, icmi_array

    data_transpose = data.transpose()
    data_shape = data_transpose.shape[0]
    icmi_array = np.zeros((data_shape, data_shape))
    np.fill_diagonal(icmi_array,np.inf)

    for i in range(0, data_shape - 1):
        for j in range(i+1, data_shape):
            icmi = maximal_information_compresseion_index(data_transpose[i],
                                                          data_transpose[j])
            icmi_array[i][j]= icmi
            icmi_array[j][i]= icmi

    return icmi_array

# cpdef np.ndarray[INT, ndim=1] SelectAtributes(np.ndarray[INT, ndim=2] icmi_array, int k):
#
#     cdef:
#         list selected_atributes = []
#         unsigned int i = 0, len_icmi_array, k_aux, index, aux
#         float l, c
#         np.ndarray sa
#
#     print "cython"
#     len_icmi_array = len(icmi_array)
#     while True:
#         k_aux = 0
#         if k > len_icmi_array:
#             k = len_icmi_array -1
#
#         # find index where icmi is min
#         index = icmi_array.argmin()
#         if index == 0:
#             break
#
#         l = index/len_icmi_array
#         c = index%len_icmi_array
#
#         selected_atributes.append(l)
#
#         while k_aux < k:
#             aux = icmi_array[l].argmin()
#             icmi_array[:,aux].fill(np.inf)
#             icmi_array[aux,:].fill(np.inf)
#             icmi_array[l][aux] = np.inf
#             k_aux += 1
#
#         icmi_array[:,l].fill(np.inf)
#         icmi_array[l,:].fill(np.inf)
#
#     selected_atributes.sort()
#     sa = np.array(selected_atributes)
#     return sa
