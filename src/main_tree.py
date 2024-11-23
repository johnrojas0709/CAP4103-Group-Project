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
from sklearn.tree import DecisionTreeClassifier

'''
    <END SET UP>
'''

'''
    Load facial landmarks (5 or 68)
'''

X = np.load("X-68-Caltech.npy")
y = np.load("y-68-Caltech.npy")
num_identities = y.shape[0]

'''
    Transform landmarks into features
'''
def eucledian(X):
    features = []
    for k in range(num_identities):
        person_k = X[k]
        features_k = []
        for i in range(person_k.shape[0]):
            for j in range(person_k.shape[0]):
                if i != j:
                    p1 = person_k[i,:]
                    p2 = person_k[j,:]      
                    features_k.append( math.sqrt( (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 ) )
        features.append(features_k)
    return np.array(features)


def angles(X, num_identities):
    features = []
    for k in range(num_identities):
        person_k = X[k]
        features_k = []
        for i in range(1, person_k.shape[0] - 1):
                p1 = person_k[i - 1, :]
                p2 = person_k[i, :]
                p3 = person_k[i + 1, :]
                angle = math.atan2(p3[1] - p2[1], p3[0] - p2[0]) - math.atan2(p1[1] - p2[1], p1[0] - p2[0])
                features_k.append(abs(angle))
        features.append(features_k)
    return np.array(features)

def relative_distance(X, num_identities):
    features = []
    for k in range(num_identities):
        person_k = X[k]
        features_k = []
        for i in range(1, person_k.shape[0], 2):
            if i + 1 < person_k.shape[0]:
                p1 = person_k[i - 1, :]
                p2 = person_k[i, :]
                distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                features_k.append(distance)
        features.append(features_k)
    return np.array(features)

#feature representation method
#features = eucledian(X)
features = angles(X, num_identities)
#features = relative_distance(X, num_identities)


''' 
    Create an instance of the classifier

'''

#depth = [1, 3, 5, 7]
#sample_split = [2, 4, 6]
#sample_leaf = [1, 3, 5]

# classifier
'''
for max_depth in depth:
    for min_samples_split in sample_split:
        for min_samples_leaf in sample_leaf:
            clf = DecisionTreeClassifier(max_depth=max_depth, 
                                         min_samples_split=min_samples_split, 
                                         min_samples_leaf=min_samples_leaf)
            '''
clf = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5)
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
print("Num correct = %d, Num incorrect = %d, Accuracy = %0.4f" 
  % (num_correct, num_incorrect, num_correct/(num_correct+num_incorrect))) 