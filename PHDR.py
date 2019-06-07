import os
import sys, getopt
from sklearn.model_selection import train_test_split
from Classifier import Classifier
from FeatureSelection import FeatureSelection
import argparse


def build_dataset(images_path='persian_digit/', fe="hog"):
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    final_path = {}
    X = []
    Y = []
    for f in files:
        tmp_list = [os.path.join(f, p) for p in sorted(os.listdir(f))]
        final_path[f[14:]] = tmp_list
        # print(final_path)
        for key, value in final_path.items():
            # print((key))
            for path in value:
                X.append(FeatureSelection().choose(function=fe, image_path=path, size=50, n_feature=5))
                Y.append(key)
    return X, Y


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--classifier", help="knn parzen bayes mlp rbf rforest dtree svm")
    parser.add_argument("-f", "--featueselect", help="pca hog resize")
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0')
    args = parser.parse_args()
    if args.classifier:
        print("classifier is ", args.classifier)
        classifier = args.classifier
    if args.featueselect:
        print("feature selection  is ", args.featueselect)
        feature_selector = args.featueselect

    X, Y = build_dataset(fe=feature_selector)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10)
    cls = Classifier().choose(classifier, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main(sys.argv[1:])
