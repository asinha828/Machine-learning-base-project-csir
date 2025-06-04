import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

DATASET_DIR = 'datasets'
IMAGE_SIZE = (100, 100)  # Resize for simplicity

def extract_faces_from_directory(dataset_dir):
    faces = []
    labels = []
    for person_name in os.listdir(dataset_dir):
        person_folder = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_folder):
            continue
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(gray, IMAGE_SIZE)
            faces.append(face.flatten())  # Use raw pixel features
            labels.append(person_name)
    return np.array(faces), np.array(labels)

# Load dataset
X, y = extract_faces_from_directory(DATASET_DIR)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train SVM
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

# Save model and label encoder
joblib.dump(clf, 'face_classifier.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("Training complete. Accuracy on test set:", clf.score(X_test, y_test))
