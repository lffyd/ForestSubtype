import os
from os.path import join as pjoin

# create class Flags
class Flags:
    DATA_DIR = './dataset/'   #Dataset Storage Directory
    DATA_FILE = 'breast_1211_23900.csv'      #Dataset  breast_1211_23900.csv、breast_2133_20000.csv
    standardization_way = 1    # Standardized approach： （1：min-max，2：z-score,....）
    isOneHot = 0  #is onthot，1 is,0 not is
    RESULT_DIR = './imageResults/finalResluts/'  # Visualization of results storage paths

    '''Clustering'''
    DATA_CLUSTER_DIR = './sClass/featureData/'  # top feature data subset storage directory
    DATA_CLUSTER_FILE = 'feature_Data_train_X.csv' # top feature data subset samples
    DATA_CLUSTER_FILE_y = 'feature_Data_train_Y.csv' # top feature data subset labels
    n_clusters = 12  #Number of clusters

    '''AE'''
    IN_OUT_SIZES = 201  # Number of input and output neurons
    Encoder_hidden_size = [1024, 512, 256, 128, 64, 32]  # Encoder hidden layer
    hidden_size = 2  # Number of neurons in the core layer
    Decoder_hidden_size = [32, 64, 128, 256, 512, 1024]  # Decoder hidden layer
    AE_epochs = 10000  # Number of model training rounds
    AE_batch_size = 128  # batchSize

    # get train dir
    def getTrainDir(self):
        local_dir = os.getcwd()
        path = Flags.DATA_DIR + Flags.DATA_FILE

        return pjoin(local_dir, path)


    # get clusterdata dir
    def getClusterDir(self):
        local_dir = os.getcwd()
        path = Flags.DATA_CLUSTER_DIR + Flags.DATA_CLUSTER_FILE

        return pjoin(local_dir, path)

    # get clustery dir
    def getClusterYDir(self):
        local_dir = os.getcwd()
        path = Flags.DATA_CLUSTER_DIR + Flags.DATA_CLUSTER_FILE_y

        return pjoin(local_dir, path)
