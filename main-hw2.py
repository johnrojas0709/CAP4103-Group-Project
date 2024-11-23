'''
    <START SET UP>
    Suppress warnings and import necessary libraries.
    Import code for loading data and extracting features.
'''

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import numpy as np
import math 

# k-Nearest Neighbors - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
from sklearn.neighbors import KNeighborsClassifier

'''
    <END SET UP>
'''

'''
    Load facial landmarks (5 or 68)
'''

X = np.load("X-5-Caltech.npy")
y = np.load("y-5-Caltech.npy")
num_identities = y.shape[0]

'''
    Transform landmarks into features
'''

features = []
for k in range(num_identities):
    person_k = X[k]
    features_k = []
    for i in range(person_k.shape[0]):
        for j in range(person_k.shape[0]):
            p1 = person_k[i,:]
            p2 = person_k[j,:]      
            features_k.append( math.sqrt( (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 ) )
    features.append(features_k)
features = np.array(features)

''' 
    Create an instance of the classifier
'''

#classifier
clf = KNeighborsClassifier()

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
print("Num correct = %d, Num incorrect = %d, Accuracy = %0.2f" 
      % (num_correct, num_incorrect, num_correct/(num_correct+num_incorrect))) 
    