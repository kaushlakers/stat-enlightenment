import sys
import matplotlib
import pandas as pd
import numpy as np
import statsmodels.api as sm

lowess = sm.nonparametric.lowess


def statistical_enlightenment(numNormals, numTumors):

    df=pd.read_csv('StatisticalEnlightenmentTestOne.csv', sep=',')
    colList = df.ix[:,0]
    df.drop('mirname', axis=1, inplace=True)
    cols = list(df.columns.values)

    dfNormal = df[[x for x in cols if x[0] == "P" ]]
    dfTumor = df[[x for x in cols if x[0] == "H" ]]
    normalMatrix = dfNormal.as_matrix()
    tumorMatrix = dfTumor.as_matrix()

    normals = normalMatrix[:,[i for i in range(numNormals)]]
    normalsTrans = normals.transpose()

    tumors = tumorMatrix[:,[i for i in range(numTumors)]]
    tumorsTrans = tumors.transpose()

    normals = normalMatrix[:,[0, numNormals]]
    normalsTrans = normals.transpose()
    std = np.std(normalsTrans, axis=0)
    mean = np.mean(normalsTrans, axis=0)

    stdEst = lowess(std, mean, return_sorted=False)

    diff = np.zeros((mean.shape[0], numTumors))
    for i in range(0, numTumors):
        result = np.absolute((mean - tumors[:,i]))/stdEst
        diff[:,i] = result

    dfResult = pd.DataFrame(diff, colList)
    return dfResult


def calcStatistic(normals, tumors):
    
    std = np.std(normals, axis=0)
    mean = np.mean(normals, axis=0)

    stdEst = lowess(std, mean, return_sorted=False)
    numTumors = tumors.shape[1]

    diff = np.zeros((mean.shape[0], numTumors))
    for i in range(0, numTumors):
        result = np.absolute((mean - tumors[:,i]))/stdEst
        diff[:,i] = result
    return diff

def generateTable(mean, stdev, runs):
    qVals = np.array([])
    for i in range(runs):
        normal = np.random.normal(mean, stdev, 1000)
        tumor = np.random.normal(mean, stdev, 1000)

        mean = np.mean([normal, tumor], axis = 0)
        std = np.std([normal, tumor], axis = 0)
        stdEst = lowess(std, mean, return_sorted=False)

        result = np.absolute((mean - tumor))/stdEst
        qVals = np.append(qVals, result)

    return pd.DataFrame(qVals)
        


def write_to_file(fileName, data, seps='\t', index = False):
    data.to_csv("Results/"+fileName, sep=seps, index=index)  


def main(args):
    '''
    numNormals = int(args[0])
    numTumors = int(args[1])
    outputFile = args[2]

    #dfResult = statistical_enlightenment(numNormals, numTumors)
    
    #send_mail("kaush.lakers@gmail.com", ["krishnankaushik.1@osu.edu"], "Results", "This is a test",
    #   files=[outputFile])
    '''
    dFrame = generateTable(0.5, 0.2, 100)
    write_to_file("QValues_100000.csv", dFrame)
if __name__ == "__main__":
   main(sys.argv[1:])