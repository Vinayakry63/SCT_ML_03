import os
import zipfile
import numpy as np
from skimage.feature import hog
from skimage import color, io
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import cv2

# Define the path to your zip file and extraction folder
zip_file_path = r'D:\Task3\dogs-vs-cats.zip'  
extracted_folder_path = r'D:\Task3\dogs-vs-cats\train'

# Extract the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder_path)

# Define path to the images folder (assuming all images are in the 'train' folder without subfolders)
path_to_data = os.path.join(extracted_folder_path, 'train')  # Update based on extracted structure
images, labels = [], []

# Loop through images and assign labels based on filename pattern
for file_name in os.listdir(path_to_data):
    img_path = os.path.join(path_to_data, file_name)
    try:
        # Label based on filename: files starting with 'cat' are labeled 1, 'dog' are labeled 0
        if 'cat' in file_name.lower():
            label = 1  # Cat
        elif 'dog' in file_name.lower():
            label = 0  # Dog
        else:
            continue  # Skip files that do not match the naming pattern

        # Read image, resize, and append to lists
        img = io.imread(img_path)
        img = cv2.resize(img, (64, 64))  # Resize image to 64x64 pixels
        images.append(img)
        labels.append(label)
    except Exception as e:
        print(f"Could not read {img_path}: {e}")

# Check if any images were loaded
if len(images) == 0:
    print("No images were loaded. Please check the file paths and structure.")
else:
    print(f"Loaded {len(images)} images.")

# Verify label counts
unique, counts = np.unique(labels, return_counts=True)
label_counts = dict(zip(unique, counts))
print(f"Label counts: {{'Dog': {label_counts[0]}, 'Cat': {label_counts[1]}}}")

# Visualize the distribution of labels
plt.bar(label_counts.keys(), label_counts.values(), tick_label=['Dog', 'Cat'])
plt.title('Distribution of Labels')
plt.xlabel('Labels')
plt.ylabel('Counts')
plt.show()

# Extract HOG features
def extract_hog_features(images):
    hog_features = []
    for img in images:
        gray_img = color.rgb2gray(img)
        features = hog(gray_img, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        hog_features.append(features)
    return np.array(hog_features)

print("Extracting HOG features...")
features = extract_hog_features(images)
print("HOG features extracted.")
print(f"Extracted features shape: {features.shape}")

# Split the dataset
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(
    features, labels, images, test_size=0.2, random_state=42
)
print(f"Training on {len(X_train)} samples and testing on {len(X_test)} samples.")

# Train the SVM classifier
print("Training SVM classifier...")
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
print("Training complete.")

# Predict on the test set
print("Predicting on test set...")
y_pred = svm.predict(X_test)
print("Prediction complete.")

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['Dog', 'Cat']))

# Function to show images with true and predicted labels
def show_images(images, true_labels, predicted_labels, num_images=10):
    plt.figure(figsize=(20, 10))
    for i in range(min(num_images, len(images))):
        plt.subplot(2, 5, i + 1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR))  # Correct color format for cv2
        plt.title(f"True: {'Dog' if true_labels[i] == 0 else 'Cat'}\nPred: {'Dog' if predicted_labels[i] == 0 else 'Cat'}")
        plt.axis('off')
    plt.show()

# Show sample images with predictions
show_images(images_test, y_test, y_pred, num_images=10)
