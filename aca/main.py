# -*- coding: utf-8 -*-
import numpy as np
from scitools.pyreport.options import verbose_execute

from k_modes import KModes

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import metrics
from sklearn.metrics import silhouette_score

from util import Open_File, Mount_CSV



def main():
    file_path = "../test.csv"
    # file_path = "/home/bertozo/Bertozo/TFG/Testes/resultados_amostragem_classes_textos/SCY-cluster_m/discover/discover.data"
    # file_path = "/home/bertozo/Bertozo/TFG/Testes/Nova pasta/2_gram_20_newsgroups_mini/discover/discover.csv"
    # file_path = "/home/bertozo/Bertozo/TFG/Testes/resultados_amostragem_classes_textos/20_newsgroups_mm/discover/20_newsgroups.csv"
    test_name = "test"
    np_array, header, numeric_data = Open_File(file_path)

    a = KModes()

    a = KModes(n_clusters=3,verbose=True)
    a.fit(numeric_data)



if __name__ == "__main__":
    main()