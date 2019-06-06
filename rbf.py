from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def rbf(X_train, X_test, y_train, y_test):
    clf = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0), warm_start=True,n_jobs=-1)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("confusion_matrix:\n\n", confusion_matrix(y_test, y_pred))
    print("classification_report:\n\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
