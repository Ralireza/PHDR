<p align="center"><img width=40% src="https://github.com/ralireza/phdr/blob/master/media/logo.png"></p>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

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
