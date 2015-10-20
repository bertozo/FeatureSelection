# -*- coding: utf-8 -*-
import numpy as np

from k_modes import KModes

from util import Open_File, Mount_CSV
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from util import Open_File, Mount_CSV


def main():
    file_path = "ionosphere.csv"
    test_name = "test"

    np_array, header, numeric_data = Open_File(file_path)

    a = KModes()
    a = KModes(n_clusters=11,verbose=False)

    a.fit(numeric_data)
    print a.cluster_modes_
    Mount_CSV(a.cluster_modes_, np_array, header, test_name)
    print "FINISH"


if __name__ == "__main__":
    main()