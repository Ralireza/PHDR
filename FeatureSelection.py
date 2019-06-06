import cv2
import os
from skimage.feature import hog
from sklearn.decomposition import PCA


class FeatureSelection:
    def __init__(self):
        print("\n----------------------------------------------------------")
        print("--------------P-R-O-C-E-S-S-I-N-G--D-A-T-A----------------")
        print("----------------------------------------------------------\n")

    def fe_resize_normalization(image_path, label, size):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
        _, im_bw = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        flatten_feature = list(im_bw.flatten())
        return flatten_feature

    def fe_hog(image_path, label, size):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)

        _, im_bw = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        fd, hog_image = hog(im_bw, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualize=True, multichannel=False)

        im_bw = list(hog_image.flatten())
        return im_bw

    def fe_pca(image_path, label, size, n_feature):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
        _, im_bw = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # print("bw:    ", len(list(im_bw.flatten())))
        pca = PCA(n_components=n_feature)
        pca.fit(im_bw)
        X = pca.transform(im_bw)
        flatten_feature = list(X.flatten())
        # print("PCA:    ", len(flatten_feature))
        return flatten_feature
