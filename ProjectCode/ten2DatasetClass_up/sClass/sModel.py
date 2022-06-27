from sklearn.ensemble import RandomForestClassifier as rf

def randomForest(max_depth=14):
    ####default max_depth=14,n_estimators=100----------####
    clf = rf(max_depth=max_depth,
             # n_estimators = n_estimators,
             random_state=0)
    return clf