# -*- coding: utf-8 -*-

import numpy as np
from setuptools.command.saveopts import saveopts
from setuptools.command.test import test
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
        icmi_array = np.load(load_icmi_matrix, mmap_mode='c')
        print "Load icmi OK *"
    else:
        print "Teste"
        t = time()
        icmi_array = _mmp.Compute_ICMI(numeric_data)
        t = time()-t
        print "Compute icmi ok, time: %f" % t

    if save_icmi_matrix and load_icmi_matrix == False:
        np.save(save_icmi_matrix + ".npy", icmi_array)
        print "Save icmi OK"

    print "Start Select Atributes:"
    # selected_atributes = _mmp.SelectAtributes_x(icmi_array, k)
    selected_atributes = SelectAtributes(icmi_array, k)
    Mount_CSV(selected_atributes, numpy_array, original_headers, test_name)

    print "k = ", k
    print "Num selected atributes: %d" % len(selected_atributes)
    print "Num atributes: %d" % (len(original_headers)-2)

    print "FINISH"

def SelectAtributes(icmi_array, k):

    selected_atributes = []
    i = 0
    mask = np.array([True]*icmi_array.shape[0])

    len_icmi_array = icmi_array.shape[0]
    k_aux = 0

    # k_neighbors = np.argsort(icmi_array)[:,:k+1]
    # kth = k_neighbors[:,k]
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


def main():
    #load_file = "../test.csv"
    #test_name = "teste"


    # load_file = "/home/toro/bds/20newsgroups.csv"
    # test_name = "20newsgroups"
    # load_icmi_matrix = "20newsgroups_icmi.npy"

    # load_file = "/home/toro/bds/ohsumed.csv"
    # test_name = "ohsumed"
    # load_icmi_matrix = "ohsumed_icmi.npy"

    load_file = "/home/toro/bds/SCY-gene.csv"
    test_name = "scy_gene"
    load_icmi_matrix = "/home/toro/testes/metricas_numpy/scy_gene_icmi.npy"

    save_icmi_matrix = test_name
    k = 1
    test_name += "_k" + str(k)
    MMP(load_file, test_name=test_name, k=k, load_icmi_matrix=load_icmi_matrix, save_icmi_matrix=False)
    print "Experimento: ", test_name, "OK"

if __name__ == "__main__":
    main()
