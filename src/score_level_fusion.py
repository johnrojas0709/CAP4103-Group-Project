import os
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.svm import SVC as svm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier as ORC
import pandas as pd

# Function to save images to the directory if they do not exist
def save_images_if_not_exists(image_directory, save_directory):
    extensions = ('jpg', 'png', 'gif', 'jpeg', 'bmp')  # Define valid image extensions
    files = os.listdir(image_directory)

    # Create the save directory if it does not exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        print(f"Directory {save_directory} created.")
    
    for file in files:
        if file.endswith(extensions):  # Process only image files
            img_path = os.path.join(image_directory, file)
            save_path = os.path.join(save_directory, file)

            if not os.path.exists(save_path):  # Save the image only if it doesn't exist in the save directory
                img = cv2.imread(img_path)
                if img is not None:
                    cv2.imwrite(save_path, img)  # Save the image to the save directory
                else:
                    pass
            else:
                pass
    
    print(f"Finished saving images. Images saved to {save_directory}.")

# Function to load images from the directory
def load_images_from_directory(save_directory):
    X, y = [], []
    extensions = ('jpg', 'png', 'gif', 'jpeg', 'bmp')  # Define valid image extensions
    files = os.listdir(save_directory)
    
    for file in files:
        if file.endswith(extensions):
            img_path = os.path.join(save_directory, file)
            img = cv2.imread(img_path)
            if img is not None:
                X.append(img)
                label = file.split('_')[0]  # Extract the label (first part of the filename)
                y.append(label)
    
    print(f"Loaded {len(X)} images with labels from {save_directory}.")
    return X, y

# Function to compute distances between facial landmarks
def compute_distances(points):
    dist = []
    for i in range(points.shape[0]):
        for j in range(points.shape[0]):
            p1 = points[i, :]
            p2 = points[j, :]
            dist.append(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))
    return dist

# Function to extract facial landmarks
def get_landmarks(images, labels, num_coords=68):
    detector = dlib.get_frontal_face_detector()
    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(predictor_path)
    landmarks, new_labels = [], []
    for idx, (img, label) in enumerate(zip(images, labels)):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_faces = detector(gray_img, 1)
        if len(detected_faces) == 0:
            continue
        for d in detected_faces:
            new_labels.append(label)
            points = np.array([[p.x, p.y] for p in predictor(gray_img, d).parts()])
            landmarks.append(compute_distances(points))
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1} images.")
    print(f"Extracted {len(landmarks)} total landmarks.")
    return np.array(landmarks), np.array(new_labels)

# Function to apply PCA for dimensionality reduction
def apply_pca(X, n_components=50):
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    print(f"PCA output dimensions: {X_reduced.shape}")
    return X_reduced

# Function to evaluate performance and plot the DET curve
def performance(gen_scores, non_gen_scores, thresholds, plot_title):
    far, frr = [], []
    for t in thresholds:
        tp = sum(1 for g in gen_scores if g >= t)
        fn = sum(1 for g in gen_scores if g < t)
        fp = sum(1 for i in non_gen_scores if i >= t)
        tn = sum(1 for i in non_gen_scores if i < t)
        far.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
        frr.append(fn / (fn + tp) if (fn + tp) > 0 else 0.0)
    plt.figure()
    plt.plot(far, frr, label="DET Curve")
    plt.xlabel("False Accept Rate (FAR)")
    plt.ylabel("False Reject Rate (FRR)")
    plt.title(plot_title)
    plt.legend()
    plt.grid()
    plt.show()

# Function to plot the score distribution of genuine and non-genuine scores
def plot_score_dist(genuine_scores, non_genuine_scores, plot_title):
    plt.figure()
    plt.hist(genuine_scores, color='green', lw=2, histtype='step', hatch='//', label='Genuine Scores')
    plt.hist(non_genuine_scores, color='red', lw=2, histtype='step', hatch='\\', label='Non-Genuine Scores')
    plt.xlim([-0.05, 1.05])
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend(loc='upper left', fontsize=12)
    plt.xlabel('Matching Score', fontsize=15, weight='bold')
    plt.ylabel('Score Frequency', fontsize=15, weight='bold')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(f'Score Distribution Plot\nSystem {plot_title}', fontsize=15, weight='bold')
    plt.show()