#author：fyd
import numpy as np
import pandas as pd
import altair as alt
import sklearn
from matplotlib import pyplot as plt
from numpy import unique
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from flag import Flags
from sklearn.metrics import accuracy_score, pairwise_distances_argmin_min
import altair_viewer
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier as knn

from sClass.sModel import randomForest
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.cluster import AgglomerativeClustering
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from itertools import cycle
import matplotlib.cm as cm
from collections import Counter

if __name__ == '__main__':
    
    data = pd.read_csv("C:/Users/Administrator/ten2DatasetClass/comparison/data/sparsewKmdata.csv",index_col=0)
    data = data.sort_values(by="class")

    results = data.iloc[:,0:2].values
    color = data.iloc[:,2].values
    TColor = data.iloc[:,3].values

    fig = plt.figure(figsize=(10,16))

    '''Clustering grouping chart'''
    legendIndexs = [0]
    for i in range(0, Flags.n_clusters - 1):
        legendIndexs.append(legendIndexs[i] + len(color[color == i]))

    colors = iter(cm.jet(np.linspace(0, 1, Flags.n_clusters)))
    ax1 = fig.add_subplot(211)
    for i in range(1, len(legendIndexs)):
        ax1.scatter(results[legendIndexs[i - 1]:legendIndexs[i], 0], results[legendIndexs[i - 1]:legendIndexs[i]:, 1],
                    color=next(colors), cmap="jet", edgecolor="None", alpha=0.35)
    ax1.scatter(results[legendIndexs[Flags.n_clusters - 1]:, 0], results[legendIndexs[Flags.n_clusters - 1]:, 1],
                color=next(colors), cmap="jet", edgecolor="None", alpha=0.35)
    ax1.set_title('sparsewkm clustering plot')
    ax1.legend(range(0, Flags.n_clusters), loc='best')

    '''Major category distribution map'''
    tLabelIndexs = []
    tLabelNum = LabelEncoder().fit_transform(TColor)
    for i in range(0, unique(TColor).size):
        tLabelIndexs.append(np.where(tLabelNum == i))

    colors2 = iter(cm.jet(np.linspace(0, 1, unique(TColor).size)))
    ax2 = fig.add_subplot(212)

    for i in range(0, len(tLabelIndexs)):
        ax2.scatter(results[tLabelIndexs[i], 0], results[tLabelIndexs[i], 1],
                    color=next(colors2), cmap="jet", edgecolor="None", alpha=0.35)
    ax2.set_title('SparsewKm clustering-PAM50 distribution')
    ax2.legend(unique(TColor), loc='best')

    plt.show()