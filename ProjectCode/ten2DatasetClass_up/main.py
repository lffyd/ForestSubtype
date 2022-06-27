#author：fyd

from MainSteps import drawingData, resultsVisualization, toCluster, AE_Study
from flag import Flags
from processData.processData import load_biology_data, load_cluster_data, load_cluster_y
from pathlib import Path

'''
Clustering 200-dimensional feature data
'''
if __name__ == '__main__':
    print("Loading feature data in...")
    data = load_cluster_data(Flags)
    print("Loading feature tags in...")
    Y = load_cluster_y(Flags)

    # Check if it has been trained based on the public dataset.
    my_file = Path("model/AE_model.h5")
    if my_file.is_file():
        # exist
        pass
    else:
        # does not exist
        # AE unsupervised
        AE_Study(data, Flags)

    #to cluster
    X_tsne, kmeansPredict = toCluster(data)

    #results Visualization
    resultsVisualization(X_tsne,kmeansPredict,Y)