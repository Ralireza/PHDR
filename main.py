import os
import sys, getopt
from sklearn.model_selection import train_test_split
from Classifier import Classifier
from FeatureSelection import FeatureSelection


def build_dataset(images_path='persian_digit/', feature_selector="hog"):
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
                X.append(FeatureSelection(feature_selector, path, 50, 5))
                Y.append(key)
    return X, Y


def main(argv):
    classifier = ''
    feature_selector = ''
    try:
        opts, args = getopt.getopt(argv, "hc:f:", ["classification=", "featureselection="])
    except getopt.GetoptError:
        print('main.py -c <classifier> -f <feature-selector>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-c", "--classification"):
            classifier = arg
        elif opt in ("-f", "--featureselection"):
            feature_selector = arg

    X, Y = build_dataset(feature_selector=feature_selector)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10)
    cls = Classifier(classifier, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main(sys.argv[1:])
