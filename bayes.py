from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB


def bayes(x_train, x_test, y_train, y_test):
    # Create a Gaussian Classifier
    gnb = GaussianNB()

    # Train the model using the training sets
    gnb.fit(x_train, y_train)

    # Predict the response for test dataset
    y_pred = gnb.predict(x_test)
    print("confusion_matrix:\n\n", confusion_matrix(y_test, y_pred))
    print("classification_report:\n\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
