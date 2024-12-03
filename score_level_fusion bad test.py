import os
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC as SVM
from sklearn.multiclass import OneVsRestClassifier as ORC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Load images and labels
def load_images_from_directory(image_directory):
    X, y = [], []
    extensions = ('jpg', 'png', 'jpeg', 'bmp')
    for idx, file in enumerate(os.listdir(image_directory)):
        if file.endswith(extensions):
            img_path = os.path.join(image_directory, file)
            img = cv2.imread(img_path)
            if img is not None:
                X.append(img)
                y.append(file.split('_')[0])
    print(f"Loaded {len(y)} images...")
    return X, y

# 2. Extract distances between facial landmarks
def compute_distances(points):
    return [
        np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        for i, p1 in enumerate(points) for j, p2 in enumerate(points)
    ]

# 3. Extract facial landmarks
def get_landmarks(images, labels):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    landmarks, new_labels = [], []
    for idx, (img, label) in enumerate(zip(images, labels)):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        if len(faces) == 0:
            continue
        for face in faces:
            points = np.array([[p.x, p.y] for p in predictor(gray, face).parts()])
            landmarks.append(compute_distances(points))
            new_labels.append(label)
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} facial landmarks...")
    return np.array(landmarks), np.array(new_labels)

# 4. Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title(title)
    plt.show()

# 5. Compute FAR, FRR, and TAR
def compute_rates(gen_scores, imp_scores, thresholds):
    far, frr, tar = [], [], []
    for t in thresholds:
        tp = sum(1 for score in gen_scores if score >= t)
        fn = len(gen_scores) - tp
        fp = sum(1 for score in imp_scores if score >= t)
        tn = len(imp_scores) - fp

        # Avoid division by zero
        if (fp + tn) > 0:
            far.append(fp / (fp + tn))
        else:
            far.append(0)

        if (fn + tp) > 0:
            frr.append(fn / (fn + tp))
        else:
            frr.append(1)

        if (tp + fn) > 0:
            tar.append(tp / (tp + fn))
        else:
            tar.append(0)
    
    return far, frr, tar


def plot_score_distribution(gen_scores, imp_scores, far, frr, plot_title):
    # Plot Genuine vs Impostor score distribution
    plt.figure(figsize=(10, 6))
    plt.hist(gen_scores, bins=50, color='green', alpha=0.7, label='Genuine Scores')
    plt.hist(imp_scores, bins=50, color='red', alpha=0.7, label='Impostor Scores')
    plt.legend()
    plt.title(f'Score Distribution: {plot_title}', fontsize=14)
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()

    # Plot FAR (False Acceptance Rate)
    plt.figure(figsize=(10, 6))
    plt.plot(far, label='FAR', color='blue')
    plt.title(f'False Acceptance Rate (FAR): {plot_title}', fontsize=14)
    plt.xlabel('Threshold')
    plt.ylabel('FAR')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot FRR (False Rejection Rate)
    plt.figure(figsize=(10, 6))
    plt.plot(frr, label='FRR', color='orange')
    plt.title(f'False Rejection Rate (FRR): {plot_title}', fontsize=14)
    plt.xlabel('Threshold')
    plt.ylabel('FRR')
    plt.legend()
    plt.grid()
    plt.show()

def main():
    """
    Dataset_configurations:
        1: Images with good lighting, trained with 90% training and 10% testing split
        2: Images with original lighting, trained with 90% training and 10% testing split
        3: Images with bad lighting for training, and images with good lighting for testing
        4: Images with good lighting for training, and images with bad lighting for testing
    """
    dataset_configurations = [
        {"split": True, "directory": "original images good lighting"},
        {"split": True, "directory": "original images"},
        {"split": False, "train": "original images bad lighting", "test": "original images good lighting"},
        {"split": False, "train": "original images good lighting", "test": "original images bad lighting"},
    ]

    for config in dataset_configurations:
        if config["split"]:
            dataset_directory = config["directory"]
            X_raw, y = load_images_from_directory(dataset_directory)
            class_names = sorted(set(y))  # Define class names dynamically
            split_index = int(len(X_raw) * 0.9)
            X_train_raw, X_test_raw = X_raw[:split_index], X_raw[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]
        else:
            X_train_raw, y_train = load_images_from_directory(config["train"])
            X_test_raw, y_test = load_images_from_directory(config["test"])
            class_names = sorted(set(y_train + y_test))  # Combine for both sets

        # Extract landmarks
        X_train, y_train = get_landmarks(X_train_raw, y_train)
        X_test, y_test = get_landmarks(X_test_raw, y_test)

        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Classifiers
        clf_knn = ORC(KNN()).fit(X_train, y_train)
        clf_svm = ORC(SVM(probability=True)).fit(X_train, y_train)
        clf_rf = ORC(RandomForestClassifier()).fit(X_train, y_train)

        # Predictions and scores
        scores_knn = clf_knn.predict_proba(X_test)
        scores_svm = clf_svm.predict_proba(X_test)
        scores_rf = clf_rf.predict_proba(X_test)
        fused_scores = (scores_knn + scores_svm + scores_rf) / 3  # Average the scores
        y_pred = np.argmax(fused_scores, axis=1)

        # Map labels to indices
        label_to_index = {label: idx for idx, label in enumerate(class_names)}
        y_test_numeric = np.array([label_to_index[label] for label in y_test])

        # Count correct and incorrect guesses
        correct_guesses = np.sum(y_pred == y_test_numeric)
        incorrect_guesses = len(y_test_numeric) - correct_guesses
        accuracy_percentage = (correct_guesses / len(y_test_numeric)) * 100

        print(f"Correct guesses: {correct_guesses}")
        print(f"Incorrect guesses: {incorrect_guesses} out of {len(y_test_numeric)}")
        print(f"Accuracy: {accuracy_percentage:.2f}%")

        # Plot confusion matrix
        plot_confusion_matrix(y_test_numeric, y_pred, class_names, title="Fused Classifier Confusion Matrix")

        # Performance evaluation
        genuine_scores = [score[y_true] for score, y_true in zip(fused_scores, y_test_numeric)]
        impostor_scores = [score.max() for score, y_true in zip(fused_scores, y_test_numeric) if score.argmax() != y_true]
        thresholds = np.linspace(0, 1, 100)
        far, frr, tar = compute_rates(genuine_scores, impostor_scores, thresholds)
        plot_score_distribution(genuine_scores, impostor_scores, far, frr, "Fused Classifier")

if __name__ == "__main__":
    main()
