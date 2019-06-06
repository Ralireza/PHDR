from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt


def knn(x_train, x_test, y_train, y_test):
    error = []
    best_k = dict()

    # Calculating error for K values between 1 and 20
    for i in range(1, 20):
        knn = KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
        knn.fit(x_train, y_train)
        pred_i = knn.predict(x_test)
        error.append(np.mean(pred_i != y_test))
        best_k[i] = np.mean(pred_i != y_test)

    best_k = sorted(best_k.items(), key=lambda k: k[1])[0][0]
    classifier = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print("confusion_matrix:\n\n", confusion_matrix(y_test, y_pred))
    print("classification_report:\n\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 20), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    plt.show()
