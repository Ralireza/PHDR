# PHDR
Persian Handwriting Digit Recognition

## Classifier
- knn
- parzen-window
- bayes
- mlp
- rbf
- svm
- random forest
- decision tree

## Feature Extractor
- PCA
- HOG
- RESIZE

## Usage
```
python3 PHDR.py -c [classifier] -f [feature-selector]
```

### Help
```
 -h, --help            show this help message and exit
  -c CLASSIFIER, --classifier CLASSIFIER
                        knn parzen bayes mlp rbf rforest dtree svm
  -f FEATUESELECT, --featueselect FEATUESELECT
                        pca hog resize
  -v, --version         show program's version number and exit
```
