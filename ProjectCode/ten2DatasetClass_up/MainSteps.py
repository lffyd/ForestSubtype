#author：fyd

import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from numpy import unique
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from flag import Flags
from sklearn.metrics import accuracy_score
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
import matplotlib.cm as cm
from collections import Counter

from processData.preVisualizationTools import getDatasByClusterNum

from usClass.usModel import AE


def AE_Study(data,Flags):
    '''
    Automatic encoder extraction of features
    :param data: Top 200 data
    :param Flags: Hyperparameter class
    :return: None
    '''

    print("AE....")

    AE_Model = AE(Flags.IN_OUT_SIZES,Flags.Encoder_hidden_size,Flags.hidden_size,Flags.Decoder_hidden_size,Flags.IN_OUT_SIZES) #201,[1024,512,256,64,32],12,[32,64,256,512,1024],201
    AE_Model.summary()
    AE_Model.compile(optimizer=tf.optimizers.Adam(),
                     loss=tf.losses.mean_squared_error,
                     metrics=['acc'])
    history = AE_Model.fit(data,data,
                 epochs=Flags.AE_epochs,
                 batch_size=Flags.AE_batch_size,
                 shuffle=True,
                 validation_data=(data,data))
    historyy = history.history
    '''
    save model
    '''
    print("save model....")
    AE_Model.save('./model/AE_model.h5')

def drawingData(X,kmeansPredict,Y):
    '''
    Preparing data for visualization
    :param X: Point-to-point
    :param kmeansPredict: k-means clustering predicted values
    :param Y: Original rough classification
    :return:
        results: Point-to-point
        preColor: Color values for subtype classification
        TColor: Roughly classified color values
    '''
    results = pd.DataFrame(columns={"x":"","y":"",'clusterNum':""},index=[0])

    for i in range(0,Flags.n_clusters):
        #Get the corresponding clustering data
        data = getDatasByClusterNum(X,kmeansPredict,Y,i)
        temp = pd.DataFrame(columns={"x":"","y":""},index=[0])
        #Iteration Data
        for row in data.itertuples():
            x = getattr(row,'x')
            y = getattr(row,'y')
            clusterNum = getattr(row,'clusterNum')
            tLabel = getattr(row,'tLabel')

            temp = temp.append({'x': x, 'y':y, 'clusterNum':clusterNum,'tLabel':tLabel}, ignore_index=True)

        results = pd.concat([results, temp],axis=0)

    results = results.dropna()
    results.index = range(len(results))
    preColor = results.iloc[:,2].T.values
    TColor = results.iloc[:, 3].T.values
    results = results.iloc[:, 0:2].values
    return results,preColor,TColor,

def toCluster(data):
    '''
    Perform clustering
    :param data: Top 200 data
    :return:
        X_tsne: Use tsne to convert high-dimensional streaming data to normal data
        kmeansPredict: k-means clustering predicted values
    '''
    # '''
    # Recovery Model
    # '''
    new_model = tf.keras.models.load_model('model/AE_model.h5')
    new_model = tf.keras.models.Sequential(new_model.layers[:7])
    new_model.build(input_shape=(data.shape[0], 201))
    new_model.summary()
    history = new_model.history
    ae_predict = new_model.predict(data)

    # 处理高维流型数据
    tsne = TSNE(verbose=0, perplexity=5, n_iter=4000,random_state=10)
    X_tsne = tsne.fit_transform(ae_predict)

    # '''kmeans++'''
    kmns = KMeans(n_clusters=Flags.n_clusters, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
                  verbose=0, random_state=1, copy_x=True, algorithm='lloyd')
    kmeansPredict = kmns.fit_predict(X_tsne)
    print("Silhouette width: %0.3f" % metrics.silhouette_score(X_tsne, kmeansPredict))
    print("DBI: %0.3f" % metrics.davies_bouldin_score(X_tsne, kmeansPredict))  # DBI

    return X_tsne,kmeansPredict

def resultsVisualization(X_tsne,kmeansPredict,Y):
    '''
    results Visualization
    :param X_tsne: Use tsne to convert high-dimensional streaming data to normal data
    :param kmeansPredict: k-means clustering predicted values
    :param Y: Original rough classification
    :return: None
    '''
    fig = plt.figure(figsize=(16, 16))

    results, color, TColor = drawingData(X_tsne, kmeansPredict, Y)
    '''Clustering grouping chart'''
    legendIndexs = [0]
    for i in range(0, Flags.n_clusters - 1):
        legendIndexs.append(legendIndexs[i] + len(color[color == i]))

    colors = iter(cm.jet(np.linspace(0, 1, Flags.n_clusters)))
    ax1 = fig.add_subplot(221)
    for i in range(1, len(legendIndexs)):
        ax1.scatter(results[legendIndexs[i - 1]:legendIndexs[i], 0], results[legendIndexs[i - 1]:legendIndexs[i]:, 1],
                    color=next(colors), cmap="jet", edgecolor="None", alpha=0.35)
    ax1.scatter(results[legendIndexs[Flags.n_clusters - 1]:, 0], results[legendIndexs[Flags.n_clusters - 1]:, 1],
                color=next(colors), cmap="jet", edgecolor="None", alpha=0.35)
    ax1.set_title('K-means clustering plot')
    ax1.legend(range(0, Flags.n_clusters), loc='best')

    '''Major category distribution map'''
    tLabelIndexs = []
    tLabelNum = LabelEncoder().fit_transform(TColor)
    for i in range(0, unique(TColor).size):
        tLabelIndexs.append(np.where(tLabelNum == i))

    colors2 = iter(cm.jet(np.linspace(0, 1, unique(TColor).size)))
    ax2 = fig.add_subplot(223)

    for i in range(0, len(tLabelIndexs)):
        ax2.scatter(results[tLabelIndexs[i], 0], results[tLabelIndexs[i], 1],
                    color=next(colors2), cmap="jet", edgecolor="None", alpha=0.35)
    ax2.set_title('clustering-PAM50 distribution')
    ax2.legend(unique(TColor), loc='best')

    '''Stacked Bar Chart'''
    x_bar = unique(color)
    y_bars = {x: np.zeros(Flags.n_clusters) for x in unique(TColor)}

    for i in x_bar:
        group = TColor[np.where(color == i)]
        count = Counter(group)
        for key, values in count.items():
            y_key = y_bars.get(key)
            y_key[int(i)] = y_key[int(i)] + values

    ax3 = fig.add_subplot(222)

    bottom_y = [0] * len(x_bar)
    for key, value in y_bars.items():
        ax3.bar(range(len(x_bar)), value, width=0.5, bottom=bottom_y)
        bottom_y = [a + b for a, b in zip(value, bottom_y)]
    ax3.set_title('Pam50 label distribution')
    ax3.set_xticks(x_bar)
    ax3.set_xlabel("cluster subtype")
    ax3.set_ylabel("Sample size")
    ax3.legend(y_bars.keys(), loc='best')
    plt.show()
    fig.savefig(Flags.RESULT_DIR+'Visualization_pdf.pdf', dpi=600,format='pdf')
    fig.savefig(Flags.RESULT_DIR+'Visualization_ps.ps', dpi=600, format='ps')
    fig.savefig(Flags.RESULT_DIR+'Visualization_tif.tif', dpi=600, format='tif')
    fig.savefig(Flags.RESULT_DIR+'Visualization_eps.eps', dpi=600, format='eps')
    fig.savefig(Flags.RESULT_DIR+'Visualization_jpeg.jpeg', dpi=600, format='jpeg')
    fig.savefig(Flags.RESULT_DIR+'Visualization_svg.svg', dpi=600, format='svg')
def purity_score(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)
