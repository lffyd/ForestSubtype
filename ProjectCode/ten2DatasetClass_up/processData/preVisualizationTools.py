from math import sqrt

import numpy as np
import pandas as pd
from numpy import random

from flag import Flags

'''Pre-visualization data processing tools'''


def getDatasByClusterNum(X_tsne,kmeansPredict,Y,clusterNum):

    clusterCoordinate_Ar = np.concatenate([X_tsne, kmeansPredict.reshape(kmeansPredict.shape[0], 1), Y.reshape(Y.shape[1], 1)],axis=1)
    clusterCoordinate_Df = pd.DataFrame(clusterCoordinate_Ar)
    clusterCoordinate_Df.columns = ['x','y','clusterNum','tLabel']
    results = clusterCoordinate_Df[clusterCoordinate_Df.loc[:,'clusterNum']==clusterNum]

    return results

def centroid(points):
    x_coords = points.iloc[:,0]
    y_coords = points.iloc[:,1]
    len = points.shape[0]
    centroid_x = sum(x_coords)/len
    centroid_y = sum(y_coords)/len
    return [centroid_x, centroid_y]

def getCenterCoordinates(X_tsne,kmeansPredict,Y):

    centerCoordinates = []

    for i in range(0,Flags.n_clusters):
        data = getDatasByClusterNum(X_tsne,kmeansPredict,Y,i)
        centercoords = centroid(data)
        centerCoordinates.append(centercoords)

    return np.array(centerCoordinates)


