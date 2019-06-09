<p align="center"><img width=40% src="https://github.com/ralireza/phdr/blob/master/media/logo.png"></p>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
[![License](https://img.shields.io/github/license/ralireza/PHDR.svg)](https://opensource.org/licenses/MIT)

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

## Report
Algorithm | Feature-Extractror | Accuracy |
--- | --- | --- 
KNN | PCA | 0.9926
KNN | HOG | 0.9889
KNN | RESIZE | 0.9933
parzen-window | PCA | 0.9926
parzen-window | HOG | 0.9889
parzen-window | RESIZE | 0.9933

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

