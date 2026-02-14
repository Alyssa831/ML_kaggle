from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def SVM_LargeScale(X, y, C_val):
    """
    Optimized SVM for large geospatial datasets.
    Uses LinearSVC which scales much better than SVC(kernel='precomputed').
    """
    # 1. Create the model
    # dual=False is recommended when n_samples > n_features
    svc = LinearSVC(C=C_val, dual=False, max_iter=10000)
    
    # 2. Fit the data directly (no manual kernel matrix needed)
    svc.fit(X, y)

    # 3. Compute accuracy
    p = svc.predict(X)
    Accuracy = accuracy_score(y, p) * 100.0
    
    return Accuracy