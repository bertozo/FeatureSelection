# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def Open_File(file_path):

    #carregando arquivo csv com cabeçalho, label na coluna 0 e call na ultima
    # coluna
    df = pd.read_csv(file_path, header = 0)

    # criando um numpy array com os dados do arquivo
    original_headers = df.columns.values
    numpy_array = df.as_matrix()
    numeric_data = df._get_numeric_data().as_matrix()

    return numpy_array, original_headers, numeric_data

def Mount_CSV(selected_atributes, numpy_array, original_headers, test_name):

    selected_atributes += 1
    selected_atributes = np.insert(selected_atributes,0,0)
    selected_atributes = np.insert(selected_atributes,len(selected_atributes),
                                   numpy_array.shape[1]-1)

    df = pd.DataFrame(numpy_array[:,selected_atributes])
    df.to_csv(test_name+".csv", header=original_headers[selected_atributes],index=False)

    # np.savetxt("results/header_"+test_name+".csv", original_headers[selected_atributes], delimiter=",", fmt='%s')
    #
    # selected_atributes = np.insert(selected_atributes,0,0)
    # selected_atributes = np.insert(selected_atributes,selected_atributes.shape[0],numpy_array.shape[1]-1)
    # selected_atributes = np.array(numpy_array[:,[selected_atributes]])
    # for i in selected_atributes:
    #     aux.append(i[0])
    # selected_atributes = np.array(aux)
    # np.savetxt("results/selected_"+test_name+".csv", selected_atributes, delimiter=",", fmt='%s')
    #
