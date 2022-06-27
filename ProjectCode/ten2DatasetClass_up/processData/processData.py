import matplotlib
import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pasta.augment import inline
# %matplotlib inline
from flag import Flags
import scipy.io as sio
from sklearn import preprocessing

'''pam50 one-hot and extract gene data'''
def preprocess(df):

    X = pd.DataFrame()
    Y = pd.DataFrame()

    #PAM50 -> ONE-HOT encoding
    pam50 = pd.get_dummies(df['PAM50'])
    pam50.columns = ['PAM50_'+str(x) for x in pam50.columns]
    Y = pd.concat([Y,pam50],axis=1)

    #x Genes
    gene = df.iloc[:, 2:]
    X = pd.concat([X, gene], axis=1)
    return (X,Y)

'''Data normalization(data[:,2:],normalization method)'''
def standardization(df,way):
    # [n_sample,n_dim] = df[:,2:].shape
    '''Take data data and convert it to array'''
    datas = df.iloc[:,2:].values  # Convert dataframe to array
    datas = datas.astype('float32')  # Defining Data Types

    if way == 1:
        #min-max standardization
        datas = preprocessing.MinMaxScaler().fit_transform(datas)
        # for i in range(n_dim):
        #     m1 = min(df[:, i])
        #     m2 = max(df[:, i])
        #     df[:, i] = (df[:, i] - m1) / (m2 - m1)
    elif  way == 2:
        #z-score standardization
        datas = preprocessing.StandardScaler().fit_transform(datas)

    dataset = pandas.DataFrame(datas)  # Reducing an array to a dataframe
    dataset.columns = df.iloc[:,2:].columns  # Name header row

    '''Re-merge pam50, sample.id, data into dataframe'''
    df = pd.concat([df.iloc[:,0:2],dataset],axis=1)

    return df

'''Load the one-hot dataset after (no splitting)'''
def load_biology_data(Flags):

    train_dir = Flags().getTrainDir()
    standardization_way = Flags.standardization_way
 #   train_size = Flags.train_size
 #   validation_size = Flags.validation_size
 #   dimension = Flags.dimension

    '''Load csv data table'''
    df_train_raw = pd.read_csv(train_dir)
    df_train_raw.head()

    '''Standardization'''
    df_train_raw = standardization(df_train_raw,standardization_way)

    if Flags.isOneHot==1:
        '''pam50 one-hot and extract gene data'''
        (X,Y) = preprocess(df_train_raw)
        return (X, Y)
    elif Flags.isOneHot==0 :
        X = df_train_raw.iloc[:, 2:]
        Y = df_train_raw.iloc[:, 0:1]
        return (X, Y)

'''Loading one-hot post-dataset (pre-segmentation)'''
def load_biology_data_split(Flags):

    train_dir = Flags().getTrainDir()
    standardization_way = Flags.standardization_way
    train_size = Flags.train_size
    validation_size = Flags.validation_size
 #   dimension = Flags.dimension

    '''Load csv data table'''
    df_train_raw = pd.read_csv(train_dir)
    df_train_raw.head()
    '''Standardization'''
    df_train_raw = standardization(df_train_raw, standardization_way)
    df_train_raw.head()
    '''pam50 one-hot and extract gene data'''
    (X,Y) = preprocess(df_train_raw)
    [n_sample,n_dim ] = X.shape

    '''Segmented Data'''
    data_train = X.iloc[0:train_size, :]
    targets_train = np.float32(Y.iloc[0:train_size, :])

    data_validation = X.iloc[train_size:train_size + validation_size, :]
    targets_validation = np.float32(Y.iloc[train_size:train_size + validation_size, :])

    data_test = X.iloc[train_size + validation_size:n_sample, :]
    targets_test = np.float32(Y.iloc[train_size + validation_size:n_sample, :])

    return (data_train, targets_train), (data_validation, targets_validation), (data_test, targets_test)

'''Load a subset of TOP features'''
def load_cluster_data(Flags):

    custer_dir = Flags().getClusterDir()

    '''Load csv data table'''
    df_custer_raw = pd.read_csv(custer_dir,index_col=0)
    df_custer_raw.head()

    return df_custer_raw

'''Load TOP feature subset Y tag'''
def load_cluster_y(Flags):
    custer_y_dir = Flags().getClusterYDir()

    '''Load csv Y-tag form'''
    df_custer_y = pd.read_csv(custer_y_dir,index_col=0)
    df_custer_y.head()

    return df_custer_y.values


# (data_train, targets_train), (data_validation, targets_validation), (data_test, targets_test) = load_biology_data_split(Flags)

#(X, y) = load_biology_data(Flags)



