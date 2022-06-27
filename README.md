# ForestSubtype

An integrated learning approach for cancer subtype classification of breast cancer using the TCGA public high-dimensional dataset.

### **Code Organization**

#### 	1.preprocessing

- idconvert.zip        #Breast cancer data from the Sangerbox 3.0 platform.   Remember to unzip the dataset when using it.
- breastPam50Classification.R    #PAM50 roughly layered.

#### 	2.ProjectCode

- comparison        #This catalogue is a comparative experiment of the sparesk method.
- dataset      		  #The catalogue is a data set.
- imageResults      #This directory contains the processing results.
- model                  #This directory contains the AE autoencoder training parameters.
- processData       #The catalogue loads data and processes it.
- sClass                  #The catalogue contains supervised models, reviews of machine learning methods and selected TOP features learned.
- usClass               #This directory contains the AE Autoencoder model.
- flag.py                 #Parameter settings
- main_jupyter.ipynb    #A priori space to guide the extraction of TOP features.
- main.py              #Discovery of cancer subtypes.
- MainSteps.py    #main.py method wrapper.

### **Requirements**

- Python
- R

### **Use the software**

1. **Data format**: filename.csv file.

2. ##### Data set description:

      X: Numerical matrix. Data other than the first two columns.Each row is a sample and each column is a gene.

      Y: Numerical vector. Column PAM50, i.e. column 0. The i-th element indicates the class to which the i-th sample belongs.

3. ##### Parameters：

   ```
   DATA_DIR   #Dataset Storage Directory
   DATA_FILE     #Dataset  breast_1211_23900.csv、breast_2133_20000.csv
   standardization_way   # Standardized approach
   isOneHot  #is onthot,1 is,0 not is
   RESULT_DIR  # Visualization of results storage paths
   
   '''Clustering'''
   DATA_CLUSTER_DIR  # top feature data subset storage directory
   DATA_CLUSTER_FILE # top feature data subset samples
   DATA_CLUSTER_FILE_y # top feature data subset labels
   n_clusters  #Number of clusters
   
   '''AE'''
   IN_OUT_SIZES  # Number of input and output neurons
   Encoder_hidden_size  # Encoder hidden layer
   hidden_size  # Number of neurons in the core layer
   Decoder_hidden_size  # Decoder hidden layer
   AE_epochs  # Number of model training rounds
   AE_batch_size  # batchSize
   ```

4. ##### Run the program：

   First, unpack the breast_1211_23900 dataset located in dataset.
   
   Then, run main_jupyter.ipynb, and finally run main.py.
