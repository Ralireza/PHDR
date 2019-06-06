import cv2
import os
from skimage.feature import hog
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from Classifier import Classifier


def build_dataset(images_path='persian_digit/',):
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
                X.append(fe_hog(path, key, 50))
                Y.append(key)
    return X, Y


X, Y = build_dataset()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10)
cls = Classifier().svm(X_train, X_test, y_train, y_test)
