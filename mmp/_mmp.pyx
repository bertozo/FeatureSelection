# -*- coding: utf-8 -*-
import cython
cimport cython

import numpy as np
cimport numpy as np
import math

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

DTYPEI = np.int
ctypedef np.int_t DTYPE_I_t


cpdef np.ndarray Compute_ICMI(np.ndarray[DTYPE_t, ndim=2] data):

    cdef:
        unsigned int i, j, data_shape
        float icmi, var_ai, var_aj, x, p
        np.ndarray[DTYPE_t, ndim=2] data_transpose, icmi_array
        np.ndarray[DTYPE_t, ndim=1] var_array

    data_transpose = data.transpose()
    data_shape = data_transpose.shape[0]

    icmi_array = np.zeros((data_shape, data_shape))
    np.fill_diagonal(icmi_array,np.inf)

    var_array = np.zeros(data_shape)

    for i in range(0, data_shape):
        var_array[i] = data_transpose[i].var()

    for i in range(0, data_shape - 1):
        print "ICMI", i
        for j in range(i+1, data_shape):
            var_ai = var_array[i]
            var_aj = var_array[j]
            p = pearson(data_transpose[i], data_transpose[j], var_ai, var_aj)
            if p != np.inf:
                x = var_ai + var_aj
                icmi = round(x - np.sqrt((x*x) - 4*var_ai*var_aj*(1 - (p*p))),5)
            else: icmi = np.inf
            icmi_array[i][j] = icmi
            icmi_array[j][i] = icmi

    return icmi_array


cdef float pearson(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] y, float var_ai, float var_aj):
    cdef:
        float r_num, r_den, r

    r_num = np.cov(x,y)[0][1]
    r_den = <float>np.sqrt(var_ai * var_aj)

    try:
        r = <float>r_num / r_den
    except:
        r = np.inf

    return r


cpdef np.ndarray[DTYPE_I_t, ndim=1] SelectAtributes_x(np.ndarray[DTYPE_t, ndim=2] icmi_array, int k):
    cdef :
        list selected_atributes = []
        unsigned int i = 0, k_aux = 0, len_icmi_array = icmi_array.shape[0], len_selected_Atributes, s , select
        np.ndarray[DTYPE_I_t, ndim=1] kth
        np.ndarray[DTYPE_I_t, ndim=2] k_neighbors
        np.ndarray mask

    mask = np.array([True]*icmi_array.shape[0])

    while True:

        len_selected_Atributes = len(selected_atributes)
        print "selecting: ", len_selected_Atributes


        if k_aux >= len_icmi_array:
            break

        if k <= 0:
            break

        k_neighbors = np.argsort(icmi_array)[:,:k+1]
        kth = k_neighbors[:,k]

        if len_icmi_array - (k_aux + k) <= 1:
            if icmi_array[mask,k_neighbors[:,0]].min() != np.inf:
                selected_atributes.append(icmi_array[mask,k_neighbors[:,0]].argmin())
            break

        # seleciona o k vizinho mais proximo (vizinho com o menor icmi)
        # quanto menor o icmi maior a correlação entre os atributos
        s = np.argmin(icmi_array[mask,kth])
        select = k_neighbors[s][0]
        selected_atributes.append(select)

        for i in k_neighbors[s]:
            icmi_array[:,i].fill(np.inf)
            icmi_array[i,:].fill(np.inf)
            k_aux += 1

        if np.min(icmi_array) == np.inf:
            break

    selected_atributes.sort()
    print selected_atributes
    return np.array(selected_atributes)