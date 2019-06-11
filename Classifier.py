from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


class Classifier:
    def __init__(self):

        print("\n----------------------------------------------------------")
        print("--------------------C-L-A-S-S-I-F-I-N-G-------------------")
        print("----------------------------------------------------------\n")

    def choose(self, algorithm, x_train, x_test, y_train, y_test):
        if algorithm == "knn":
            self.knn(x_train, x_test, y_train, y_test)
        if algorithm == "parzen":
            self.knn(x_train, x_test, y_train, y_test, is_parzen=True)
        if algorithm == "bayes":
            self.bayes(x_train, x_test, y_train, y_test)
        if algorithm == "mlp":
            self.mlp(x_train, x_test, y_train, y_test)
        if algorithm == "rbf":
            self.rbf(x_train, x_test, y_train, y_test)
        if algorithm == "svm":
            self.svm(x_train, x_test, y_train, y_test)
        if algorithm == "dtree":
            self.decision_tree(x_train, x_test, y_train, y_test)
        if algorithm == "rforest":
            self.random_forest(x_train, x_test, y_train, y_test)

    def knn(self, x_train, x_test, y_train, y_test, is_parzen=False):
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
        if is_parzen:
            classifier = KNeighborsClassifier(n_neighbors=best_k, algorithm='ball_tree', n_jobs=-1)
        else:
            classifier = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)

        self.report(y_test, y_pred)

        plt.figure(figsize=(12, 6))
        plt.plot(range(1, 20), error, color='red', linestyle='dashed', marker='o',
                 markerfacecolor='blue', markersize=10)
        plt.title('Error Rate K Value')
        plt.xlabel('K Value')
        plt.ylabel('Mean Error')
        plt.show()

    def bayes(self, x_train, x_test, y_train, y_test):
        # Create a Gaussian Classifier
        gnb = GaussianNB()

        # Train the model using the training sets
        gnb.fit(x_train, y_train)

        # Predict the response for test dataset
        y_pred = gnb.predict(x_test)
        self.report(y_test, y_pred)

    def mlp(self, X_train, X_test, y_train, y_test):
        clf = MLPClassifier(hidden_layer_sizes=(50), max_iter=10000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        self.report(y_test, y_pred)

    def rbf(self, X_train, X_test, y_train, y_test):
        clf = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0), warm_start=True, n_jobs=-1)
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        self.report(y_test, y_pred)

    def svm(self, X_train, X_test, y_train, y_test):
        clf = SVC(gamma='auto')
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        self.report(y_test, y_pred)

    def decision_tree(self, X_train, X_test, y_train, y_test):
        clf = DecisionTreeClassifier(random_state=0)
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        self.report(y_test, y_pred)

    def random_forest(self, X_train, X_test, y_train, y_test):
        clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=0)
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        self.report(y_test, y_pred)

    def report(self, y_test, y_pred):
        print("confusion_matrix:\n\n", confusion_matrix(y_test, y_pred))
        print("----------------------------------------------------------")
        print("----------------------------------------------------------\n")
        print("classification_report:\n\n", classification_report(y_test, y_pred))
        print("----------------------------------------------------------")
        print("----------------------------------------------------------\n")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("----------------------------------------------------------")
        print("----------------------------------------------------------\n")
    # TODO ok this function
    # https://blog.goodaudience.com/music-genre-classification-using-hidden-markov-models-4a7f14eb0fd4
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')