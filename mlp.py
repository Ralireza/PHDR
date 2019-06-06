from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def mlp(X_train, X_test, y_train, y_test):
    clf = MLPClassifier(hidden_layer_sizes=(10), max_iter=500)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("confusion_matrix:\n\n", confusion_matrix(y_test, y_pred))
    print("classification_report:\n\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
