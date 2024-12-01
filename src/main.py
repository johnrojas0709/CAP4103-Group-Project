from score_level_fusion import save_images_if_not_exists, load_images_from_directory, get_landmarks, apply_pca, plot_score_dist
import os
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.svm import SVC as svm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier as ORC
import pandas as pd

def main(image_directory):
    save_directory = "saved_images"  # Directory to save images if not already present
    
    # Save images to the directory if they don't exist
    save_images_if_not_exists(image_directory, save_directory)
    
    # Load images from the save directory
    X_raw, y = load_images_from_directory(save_directory)
    if len(X_raw) == 0:
        print("No images loaded.")
        return

    # Extract facial landmarks
    X_landmarks, y_landmarks = get_landmarks(X_raw, y)
    if X_landmarks.shape[0] == 0:
        print("No landmarks extracted.")
        return

    # Apply PCA for dimensionality reduction
    X_reduced = apply_pca(X_landmarks)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reduced)
    print(f"Scaled feature dimensions: {X_scaled.shape}")

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_landmarks, test_size=0.20, random_state=42)
    print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")

    # Define classifiers
    clf_knn = ORC(knn())
    clf_svm = ORC(svm(probability=True))
    clf_rf = ORC(RandomForestClassifier(n_estimators=100))

    # Train classifiers
    clf_knn.fit(X_train, y_train)
    clf_svm.fit(X_train, y_train)
    clf_rf.fit(X_train, y_train)

    # Predict scores
    scores_knn = clf_knn.predict_proba(X_test)
    scores_svm = clf_svm.predict_proba(X_test)
    scores_rf = clf_rf.predict_proba(X_test)

    # Normalize scores using Min-Max scaling to the range [0, 1]
    scaler = MinMaxScaler()
    scores_knn_norm = scaler.fit_transform(scores_knn)
    scores_svm_norm = scaler.fit_transform(scores_svm)
    scores_rf_norm = scaler.fit_transform(scores_rf)

    # Fuse the normalized scores
    fused_scores = (scores_knn_norm + scores_svm_norm + scores_rf_norm) / 3.0
    print(f"Fused scores dimensions: {fused_scores.shape}")

    # Convert fused scores into a DataFrame for easier manipulation
    classes = clf_knn.classes_
    fused_scores_df = pd.DataFrame(fused_scores, columns=classes)

    # Initialize variables to keep track of accuracy and inaccurate guesses
    correct_guesses = 0
    incorrect_guesses = 0
    genuine_scores, non_genuine_scores = [], []

    # Compare predicted labels with true labels
    for i, label in enumerate(y_test):
        predicted_label = fused_scores_df.iloc[i].idxmax()  # The predicted label is the one with the highest score
        score = fused_scores_df.iloc[i][predicted_label]
        
        if predicted_label == label:
            correct_guesses += 1
            genuine_scores.append(score)  # Genuine scores
        else:
            incorrect_guesses += 1
            non_genuine_scores.append(score)  # Non-genuine scores

    # Calculate accuracy
    accuracy = correct_guesses / len(y_test)

    # Print the results
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Inaccurate guesses: {incorrect_guesses} out of {len(y_test)}")

    # Plot score distribution of genuine and non-genuine scores
    plot_score_dist(genuine_scores, non_genuine_scores, "Fused Classifier")

if __name__ == "__main__":
    main("olivetti_faces_images")
