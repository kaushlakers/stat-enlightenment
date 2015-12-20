import sys
import matplotlib
import pandas as pd
import numpy as np
import statsmodels.api as sm

lowess = sm.nonparametric.lowess

def calc_statistic(normal, tumor):
    
    mean = np.mean([normal, tumor], axis = 0)
    std = np.std([normal, tumor], axis = 0)
    stdEst = lowess(std, mean, return_sorted=False)

    result = np.absolute((mean - tumor))/stdEst
    
    return result

#filename, col to use as rowname, col to use as values
def read_file(filename, name, value):
    if filename.split('.')[1] == 'csv':
        df=pd.read_csv('Data/'+filename, sep=',')
    else:
        df = pd.read_table('Data/'+filename)
    df_filtered = df[[x for x in [name, value]]]
    return df_filtered

def write_to_file(fileName, data, seps='\t', index = False):
    data.to_csv("Results/"+fileName, sep=seps, index=index)

def main(argv):
    norm_file, tum_file, col_name, col_value, out_file = argv[1], argv[2], argv[3], argv[4], argv[5]

    df_normal = read_file(norm_file, col_name, col_value)
    df_tumor = read_file(tum_file, col_name, col_value)

    col_name_list = df_normal[col_name].as_matrix()

    result_array = calc_statistic(df_normal[col_value].as_matrix(), df_tumor[col_value].as_matrix())

    df_result = pd.DataFrame(result_array, col_name_list)

    write_to_file(out_file, df_result, index=True)


if __name__ == "__main__":
    print sys.argv
    main(sys.argv)