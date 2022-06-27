from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import altair_viewer
import numpy as np
import pandas as pd
import altair as alt
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

'''This .py is used to test the optimal parameters of each supervised learning method'''

def randomForestAssess(X_train,y_train,X_test,y_test):

    '''Random Forest'''
    #1、Find the optimal depth and find that the optimal depth is 11
    max_depth_best = []
    for i in range(1, 20):
        clf = rf(max_depth=i, random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        max_depth_best.append([i, accuracy_score(y_test, y_pred)])
    max_depth = pd.DataFrame(max_depth_best, columns=["max_depth", "Accuracy"])

    print(max_depth)

    max_depth_chart = alt.Chart(max_depth).mark_line().encode(
        alt.X("max_depth", scale=alt.Scale(zero=False)),
        alt.Y("Accuracy", scale=alt.Scale(zero=False)),
        tooltip=["max_depth", "Accuracy"]
    ).properties(
        title="Random Forest Acuracy vs Max Depth"
    )
    max_depth_chart.show()

    #2、Selecting the optimal evaluator
    n_estimators = []
    for i in range(10, 200):
        clf = rf(max_depth=11, n_estimators=i, random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        n_estimators.append([i, accuracy_score(y_test, y_pred)])

    n_estimate = pd.DataFrame(n_estimators, columns=["n_estimator", "Accuracy"])
    estimate_chart = alt.Chart(n_estimate).mark_line().encode(
        alt.X("n_estimator", scale=alt.Scale(zero=False)),
        alt.Y("Accuracy", scale=alt.Scale(zero=False)),
        tooltip=['n_estimator', "Accuracy"]
    ).properties(
        title="Random Forest Acuracy vs n_estimator"
    )
    max_depth_chart.show | estimate_chart.show()

    print(n_estimate.loc[n_estimate["Accuracy"].idxmax()])

    # param_distribution = {
    #     # "n_estimators": np.arange(10, 200,1),
    #     "max_depth": np.arange(1, 20,1),
    #     "min_samples_split":np.arange(10,101,1),
    #     # "min_samples_leaf":np.arange(5,51,5),
    #     # "max_features":np.arange(3,11,1)
    # }
    #
    # gsc = GridSearchCV(model,
    #                    param_distribution,
    #                    cv = 10,
    #                    n_jobs = 1)
    # gsc.fit(X_train, y_train)
    # #Print the best results
    # print(gsc.best_params_)
    # print(gsc.best_score_)
    # print(gsc.best_estimator_)

def kneighborsAssess(X_train,y_train,X_test,y_test):
    '''knn'''
    neighbors = []

    for i in range(1, 20):
        kclass = knn(n_neighbors=i)
        kclass.fit(X_train, y_train)
        y_pred = kclass.predict(X_test)
        neighbors.append([i, accuracy_score(y_test, np.ravel(y_pred))])
    neighbors = pd.DataFrame(neighbors, columns=["n_neighbors", "Accuracy"])
    neighbors.head()

    alt.Chart(neighbors).mark_line().encode(
        alt.X("n_neighbors", scale=alt.Scale(zero=False)),
        alt.Y("Accuracy", scale=alt.Scale(zero=False)),
        tooltip=["Accuracy", "n_neighbors"]
    ).properties(
        title="Accuracy vs Number of Neighbors"
    )

def svmmAssess(X_train,y_train,X_test,y_test):
    '''svm'''
    svm_class = []
    for i in range(1, 10):
        clf = make_pipeline(StandardScaler(), SVC(C=i))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        svm_class.append([i, accuracy_score(y_test, np.ravel(y_pred))])
    svm_class = pd.DataFrame(svm_class, columns=["C parameter", "Accuracy"])
    svm_class.head()

    alt.Chart(svm_class).mark_line().encode(
        alt.X("C parameter", scale=alt.Scale(zero=False)),
        alt.Y("Accuracy", scale=alt.Scale(zero=False))
    ).properties(
        title="ACCURACY VS C PARAMETER SVM"
    )

    svm_model = make_pipeline(StandardScaler(), SVC(C=8))

    # Fitting Model
    svm_model.fit(X_train, y_train)

    # Obtaining prediction
    y_pred_svm = svm_model.predict(X_test)

    # Getting the Confusion Matrix
    svm_conf_mat = confusion_matrix(y_test, y_pred_svm, labels=svm_model.classes_)
    ConfusionMatrixDisplay(svm_conf_mat, display_labels=svm_model.classes_).plot()

def logisticRegressionAssess(X_train,y_train,X_test,y_test):
    #'''Logistic regression model'''
    log_class =  LogisticRegression(random_state=0, solver="saga",max_iter=200)
    log_class.fit(X_train, y_train)
    log_class.score(X_test, y_test)

    y_pred_log = log_class.predict(X_test)
    log_conf_mat = confusion_matrix(y_test, y_pred_log, labels=log_class.classes_)
    ConfusionMatrixDisplay(log_conf_mat, display_labels=log_class.classes_).plot()

def MLPClassifierAssess(X_train,y_train,X_test,y_test):
    # '''Multi-layer perceptron classifier'''

    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(1024, 512), random_state=1,
                        max_iter=1000)  # Create neural network classifier objects
    clf.fit(X_train, y_train)  # Training Model
    clf.score(X_test, y_test)  # Model Scoring

def combinationAssess(X_train,y_train,X_test,y_test):
    rf_clf = rf(max_depth=13, random_state=0)
    svm_clf = SVC(C=8)
    log_clf = LogisticRegression(random_state=0, solver="saga", max_iter=200)

    ensembl_clf = VotingClassifier(
        estimators=[("rf", rf_clf), ("log", log_clf)],
        voting="hard"
    )

    ensembl_clf.fit(X_train, y_train)
    ensembl_clf.fit_transform()
    ensembl_clf.score(X_test, y_test)


'''Average purity'''
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


