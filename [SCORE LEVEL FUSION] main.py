import os
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.svm import SVC as svm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier as ORC
import pandas as pd

# Function to load images and labels from a directory
def load_images_from_directory(image_directory):
    X, y = [], []
    extensions = ('jpg', 'png', 'gif')
    files = os.listdir(image_directory)
    for file in files:
        if file.endswith(extensions):
            img_path = os.path.join(image_directory, file)
            img = cv2.imread(img_path)
            if img is not None:
                X.append(img)
                label = file.split('_')[0]  # Extract the label (first part of the filename)
                y.append(label)
    print(f"Loaded {len(X)} images with labels.")
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

# Function to compute rates (FAR, FRR, TAR)
def compute_rates(gen_scores, imp_scores, thresholds):
    far = []
    frr = []
    tar = []
    
    for t in thresholds:
        tp = 0  # True Positives
        fp = 0  # False Positives
        tn = 0  # True Negatives
        fn = 0  # False Negatives
        
        for g_s in gen_scores:
            if g_s >= t:
                tp += 1
            else:
                fn += 1
                
        for i_s in imp_scores:
            if i_s >= t:
                fp += 1
            else:
                tn += 1
                    
        far.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
        frr.append(fn / (fn + tp) if (fn + tp) > 0 else 0.0)
        tar.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        
    return far, frr, tar

# Function to plot the score distribution
def plot_scoreDist(gen_scores, imp_scores, far, frr, plot_title):
    plt.figure()
    plt.hist(gen_scores, color='green', lw=2, histtype='step', hatch='//', label='Genuine Scores')
    plt.hist(imp_scores, color='red', lw=2, histtype='step', hatch='\\', label='Impostor Scores')
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
    plt.savefig(f'score_dist_{plot_title}.png', dpi=300, bbox_inches="tight")

# Function to evaluate performance and plot the DET curve
def performance(gen_scores, imp_scores, plot_title, num_thresholds):
    thresholds = np.linspace(0, 1, num_thresholds)
    far, frr, tar = compute_rates(gen_scores, imp_scores, thresholds)    
    plot_scoreDist(gen_scores, imp_scores, far, frr, plot_title)

# Main execution function
def main(image_directory):
    # Load images and labels
    X_raw, y = load_images_from_directory(image_directory)
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
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_landmarks, test_size=0.33, random_state=42)
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

    # Fuse scores
    fused_scores = (scores_knn + scores_svm + scores_rf) / 3.0
    print(f"Fused scores dimensions: {fused_scores.shape}")

    # Convert fused scores into a DataFrame for easier manipulation
    classes = clf_knn.classes_
    fused_scores_df = pd.DataFrame(fused_scores, columns=classes)

    # Initialize variables to keep track of accuracy and inaccurate guesses
    correct_guesses = 0
    incorrect_guesses = 0

    # Compare predicted labels with true labels
    for i, label in enumerate(y_test):
        predicted_label = fused_scores_df.iloc[i].idxmax()  # The predicted label is the one with the highest score
        if predicted_label == label:
            correct_guesses += 1
        else:
            incorrect_guesses += 1

    # Calculate accuracy
    accuracy = correct_guesses / len(y_test)

    # Print the results
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Inaccurate guesses: {incorrect_guesses} out of {len(y_test)}")

    # Evaluate performance
    thresholds = np.linspace(0, 1, 100)
    performance(gen_scores=fused_scores_df.values[:, y_test], 
                imp_scores=fused_scores_df.values[:, ~y_test],
                plot_title="Fused Classifier Performance", 
                num_thresholds=100)

if __name__ == "__main__":
    main("olivetti_faces_images")
