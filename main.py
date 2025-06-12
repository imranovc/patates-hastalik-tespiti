import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from skimage.feature import local_binary_pattern
from skimage.util import random_noise
from sklearn.naive_bayes import GaussianNB


def add_noise(image):
    noisy_image = random_noise(image, mode='gaussian', var=0.01)
    noisy_image = (noisy_image * 255).astype('uint8')
    return noisy_image

def augment_image(image):
    augmented_images = []
    rows, cols = image.shape[:2]
    for angle in [90, 180, 270]:
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated = cv2.warpAffine(image, rotation_matrix, (cols, rows))
        augmented_images.append(rotated)
    augmented_images.append(cv2.flip(image, 1))
    augmented_images.append(cv2.flip(image, 0))
    augmented_images.append(add_noise(image))
    bright_image = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
    augmented_images.append(bright_image)
    return augmented_images

def extract_color_features(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

def extract_texture_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    return hist / hist.sum()

def extract_shape_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(c, True)
        area = cv2.contourArea(c)
        return [area, perimeter, (perimeter**2) / (4 * np.pi * area)]
    return [0, 0, 0]

def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor()
    h = hog.compute(gray)
    return h.flatten()


def extract_features(image):
    color_features = extract_color_features(image)
    texture_features = extract_texture_features(image)
    shape_features = extract_shape_features(image)
    hog_features = extract_hog_features(image)
    return np.hstack([color_features, texture_features, shape_features,hog_features])

def load_data(folder_path):
    features = []
    labels = []
    for label, class_name in enumerate(["healthy", "late", "early"]):
        class_path = os.path.join(folder_path, class_name)
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))
            features.append(extract_features(img))
            labels.append(label)
            augmented_images = augment_image(img)
            for aug_img in augmented_images:
                features.append(extract_features(aug_img))
                labels.append(label)
    return np.array(features), np.array(labels)

data_folder = "dataset"
X, y = load_data(data_folder)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



rf_model = RandomForestClassifier(n_estimators=70,random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))


dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))
