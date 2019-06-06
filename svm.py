from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def svm(X_train, X_test, y_train, y_test):
    clf = SVC(gamma='auto')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("confusion_matrix:\n\n", confusion_matrix(y_test, y_pred))
    print("classification_report:\n\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
