import os
import cv2
import dlib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC as SVM
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier as ORC
import matplotlib.pyplot as plt
import pandas as pd

# Function to load images and labels
def load_images_from_directory(image_directory):
    X, y = [], []
    extensions = ('jpg', 'png', 'jpeg', 'bmp')  # Valid image extensions
    files = os.listdir(image_directory)

    for file in files:
        if file.endswith(extensions):
            img_path = os.path.join(image_directory, file)
            img = cv2.imread(img_path)
            if img is not None:
                X.append(img)
                label = file.split('_')[0]  # Extract label (first part of filename)
                y.append(label)
    
    print(f"Loaded {len(X)} images with labels from {image_directory}.")
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
def get_landmarks(images, labels):
    detector = dlib.get_frontal_face_detector()
    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(predictor_path)
    landmarks, new_labels = [], []

    for idx, (img, label) in enumerate(zip(images, labels)):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_faces = detector(gray_img, 1)
        if len(detected_faces) == 0:
            continue
        for face in detected_faces:
            points = np.array([[p.x, p.y] for p in predictor(gray_img, face).parts()])
            landmarks.append(compute_distances(points))
            new_labels.append(label)
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

# Function to plot score distribution of genuine and non-genuine scores
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

# Main function
def main():
    dataset_train = "original images good lighting"
    dataset_test = "original images bad lighting"

    # Load all training images and labels
    X_train_raw, y_train = load_images_from_directory(dataset_train)
    if len(X_train_raw) == 0:
        print("No training images loaded.")
        return

    # Load all testing images and labels
    X_test_raw, y_test = load_images_from_directory(dataset_test)
    if len(X_test_raw) == 0:
        print("No testing images loaded.")
        return

    print(f"Training images: {len(X_train_raw)}, Testing images: {len(X_test_raw)}")

    # Extract landmarks for training data
    X_train_landmarks, y_train_landmarks = get_landmarks(X_train_raw, y_train)
    if X_train_landmarks.shape[0] == 0:
        print("No landmarks extracted from training data.")
        return

    # Extract landmarks for testing data
    X_test_landmarks, y_test_landmarks = get_landmarks(X_test_raw, y_test)
    if X_test_landmarks.shape[0] == 0:
        print("No landmarks extracted from testing data.")
        return

    # Scale features directly without applying PCA
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_landmarks)
    X_test_scaled = scaler.transform(X_test_landmarks)

    print(f"Training set: {X_train_scaled.shape}, Testing set: {X_test_scaled.shape}")

    # Define classifiers
    clf_knn = ORC(KNN())
    clf_svm = ORC(SVM(probability=True))
    clf_rf = ORC(RandomForestClassifier(n_estimators=100))

    # Train classifiers
    clf_knn.fit(X_train_scaled, y_train_landmarks)
    clf_svm.fit(X_train_scaled, y_train_landmarks)
    clf_rf.fit(X_train_scaled, y_train_landmarks)

    # Predict scores
    scores_knn = clf_knn.predict_proba(X_test_scaled)
    scores_svm = clf_svm.predict_proba(X_test_scaled)
    scores_rf = clf_rf.predict_proba(X_test_scaled)

    # Normalize scores using Min-Max scaling
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

    # Evaluate predictions
    correct_guesses = 0
    incorrect_guesses = 0
    genuine_scores, non_genuine_scores = [], []

    for i, label in enumerate(y_test_landmarks):
        predicted_label = fused_scores_df.iloc[i].idxmax()
        score = fused_scores_df.iloc[i][predicted_label]

        if predicted_label == label:
            correct_guesses += 1
            genuine_scores.append(score)
        else:
            incorrect_guesses += 1
            non_genuine_scores.append(score)

    # Calculate accuracy
    accuracy = correct_guesses / len(y_test_landmarks)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Inaccurate guesses: {incorrect_guesses} out of {len(y_test_landmarks)}")

    # Plot score distribution
    plot_score_dist(genuine_scores, non_genuine_scores, "Fused Classifier")



if __name__ == "__main__":
    main()
