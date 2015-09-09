# -*- coding: utf-8 -*-

import numpy as np
from time import time
import pandas as pd
import _mmp
from util import Open_File, Mount_CSV


def MMP(load_file, test_name, k, load_icmi_matrix=False,
        save_icmi_matrix=False):

    original_headers = []

    numpy_array, original_headers, numeric_data = Open_File(load_file)
    print "Open file OK"


    ## COMENTAR/DESCOMENTAR SE JÁ TIVER O ICM_SORTED

    # Salvar ou carregar matriz dos icmi ja computados
    if load_icmi_matrix:
        icmi_array = np.load(load_icmi_matrix)
        print "Load icmi OK"
    else:
        # computar icmi
        t = time()
        icmi_array = _mmp.Compute_ICMI(numeric_data)
        t = time()-t
        print "Compute icmi ok, time: %f" % t

    if save_icmi_matrix and load_icmi_matrix == False:
        np.save(test_name + ".npy", icmi_array)
        print "Save icmi OK"


    print "Start Select Atributes:"

    selected_atributes = SelectAtributes(icmi_array, k)
    Mount_CSV(selected_atributes, numpy_array, original_headers, test_name)

    print "k = ", k
    print "Num selected atributes: %d" % len(selected_atributes)
    print "Num atributes: %d" % (len(original_headers)-2)

    print "FINISH"

def SelectAtributes(icmi_array, k):

    selected_atributes = []

    i = 0

    len_icmi_array = len(icmi_array)
    while True:
        k_aux = 0
        if k > len_icmi_array:
            k = len_icmi_array -1

        # find index where icmi is min
        index = icmi_array.argmin()
        if index == 0:
            break

        l = index/len_icmi_array
        c = index%len_icmi_array

        selected_atributes.append(l)

        while k_aux < k:
            aux = icmi_array[l].argmin()
            icmi_array[:,aux].fill(np.inf)
            icmi_array[aux,:].fill(np.inf)
            icmi_array[l][aux] = np.inf
            k_aux += 1

        icmi_array[:,l].fill(np.inf)
        icmi_array[l,:].fill(np.inf)

    selected_atributes.sort()
    return np.array(selected_atributes)


def main():
    # load_file = "../test.csv"
    # test_name = "test"

    load_file = "/home/bertozo/Área de Trabalho/RESULTADOS/20_newsgroups/00_20_newsgroups.csv"
    test_name = "MMP_k3_selected_20_newsgroups"
    k = 3
    MMP(load_file, test_name, k,save_icmi_matrix=True)

if __name__ == "__main__":
    main()
