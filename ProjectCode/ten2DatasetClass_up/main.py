#author：fyd
import sys

from MainSteps import drawingData, resultsVisualization, toCluster, AE_Study
from flag import Flags
from processData.processData import load_biology_data, load_cluster_data, load_cluster_y
from pathlib import Path

'''
Clustering 200-dimensional feature data
'''
if __name__ == '__main__':
    print("Loading feature data in...")
    data = ""
    Y = ""
    try:
        data = load_cluster_data(Flags)
    except FileNotFoundError:
        print("File:"+Flags.DATA_CLUSTER_FILE,"does not exist,Please run the main_jupyter.ipynb file first.")
        sys.exit()

    print("Loading feature tags in...")
    try:
        Y = load_cluster_y(Flags)
    except FileNotFoundError:
        print("File:"+Flags.DATA_CLUSTER_FILE_y,"does not exist,Please run the main_jupyter.ipynb file first.")
        sys.exit()
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