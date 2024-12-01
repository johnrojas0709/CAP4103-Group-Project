import os
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC as svm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier as ORC
import pandas as pd

# test different values for n_neighbors
k_vals = [1, 3, 5, 7]
for k in k_vals:
#classifier
    clf = KNeighborsClassifier(n_neighbors=k)
    
    num_correct = 0
    num_incorrect = 0
    
    #feature representation
    for i in range(0, len(y)):
        query_X = features[i, :]
        query_y = y[i]
        
        template_X = np.delete(features, i, 0)
        template_y = np.delete(y, i)
            
        # Set the appropriate labels
        # 1 is genuine, 0 is impostor
        y_hat = np.zeros(len(template_y))
        y_hat[template_y == query_y] = 1 
        y_hat[template_y != query_y] = 0
        
        # Train the classifier
        clf.fit(template_X, y_hat) 
        
        # Predict the label of the query
        y_pred = clf.predict(query_X.reshape(1,-1)) 
        
        # Get results
        if y_pred == 1:
            num_correct += 1
        else:
            num_incorrect += 1

# Print results
    print()
    print("k = %d, Num correct = %d, Num incorrect = %d, Accuracy = %0.2f" 
      % (k, num_correct, num_incorrect, num_correct/(num_correct+num_incorrect))) 