''' Imports '''
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import get_images
import get_landmarks
import performance_plots

from sklearn.multiclass import OneVsRestClassifier as ORC
from sklearn.model_selection import train_test_split
import pandas as pd


''' Import classifiers '''
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.svm import SVC as svm

''' Load the data and their labels '''
image_directory = '../Caltech Faces Dataset'
X, y = get_images.get_images(image_directory)

''' Get distances between face landmarks in the images '''
X, y = get_landmarks.get_landmarks(X, y, 'landmarks/', 5, False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

''' Matching and Decision - Classifer 1 '''
clf = ORC(knn())
clf.fit(X_train, y_train)
matching_scores_knn = clf.predict_proba(X_test)

''' Matching and Decision - Classifer 2 '''
clf = ORC(svm(probability=True))
clf.fit(X_train, y_train)
matching_scores_svm = clf.predict_proba(X_test)

''' Fuse scores '''
matching_scores = (matching_scores_knn + matching_scores_svm) / 2.0

gen_scores = []
imp_scores = []
classes = clf.classes_
matching_scores = pd.DataFrame(matching_scores, columns=classes)

for i in range(len(y_test)):    
    scores = matching_scores.loc[i]
    mask = scores.index.isin([y_test[i]])
    gen_scores.extend(scores[mask])
    imp_scores.extend(scores[~mask])
    
performance_plots.performance(gen_scores, imp_scores, 'kNN-SVM-score_fusion', 100)



    
    
