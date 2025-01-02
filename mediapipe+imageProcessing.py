!pip install mediapipe

import time
import numpy as np
import cv2
import mediapipe as mp
from mtcnn import MTCNN
import dlib
from google.colab import files
import matplotlib.pyplot as plt

# Function to adjust contrast
def adjust_contrast(image, contrast_factor):
    return cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)

# Function to crop image
def crop_image(image, x1, y1, x2, y2):
    return image[y1:y2, x1:x2]

# Function to calculate Normalized Mean Error (NME)
def calculate_nme(detected_landmarks, ground_truth_landmarks):
    distances = np.linalg.norm(detected_landmarks - ground_truth_landmarks, axis=1)
    ipd = np.linalg.norm(ground_truth_landmarks[0] - ground_truth_landmarks[1])
    return np.mean(distances) / ipd

# Detection function for MediaPipe
def detect_with_mediapipe(image):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            landmarks = [
                (lmk.x * image.shape[1], lmk.y * image.shape[0])
                for lmk in results.multi_face_landmarks[0].landmark
            ]
            return np.array(landmarks[:5])
    return None

# Evaluation function
def evaluate_model_performance(model_name, detection_function, image_array, ground_truth_landmarks):
    start_time = time.time()
    try:
        detected_landmarks = detection_function(image_array.copy())
        elapsed_time = time.time() - start_time
        if detected_landmarks is not None:
            nme = calculate_nme(detected_landmarks, ground_truth_landmarks)
            robustness_score = len(detected_landmarks)
            return elapsed_time, nme, robustness_score
        else:
            return elapsed_time, None, 0
    except Exception as e:
        print(f"Error during {model_name} detection: {e}")
        return None, None, 0

# Upload and read image
uploaded = files.upload()
image_path = list(uploaded.keys())[0]
image = cv2.imdecode(np.frombuffer(uploaded[image_path], np.uint8), cv2.IMREAD_COLOR)

if image is None:
    print("Error: Could not load the image.")
    exit()

# Step 1: Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.show()

# Step 2: Split into RGB channels
b, g, r = cv2.split(image)
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.subplot(1, 4, 2)
plt.imshow(b, cmap='gray')
plt.title('Blue Channel')
plt.subplot(1, 4, 3)
plt.imshow(g, cmap='gray')
plt.title('Green Channel')
plt.subplot(1, 4, 4)
plt.imshow(r, cmap='gray')
plt.title('Red Channel')
plt.show()

# Step 3: Adjust contrast
contrast_factor = 1.5
enhanced_contrast = adjust_contrast(image, contrast_factor)
plt.imshow(cv2.cvtColor(enhanced_contrast, cv2.COLOR_BGR2RGB))
plt.title('Enhanced Contrast')
plt.show()

# Step 4: Crop the image
x1, y1, x2, y2 = 50, 50, 300, 300
cropped_image = crop_image(image, x1, y1, x2, y2)
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.title('Cropped Image')
plt.show()

# Step 5: Detect facial landmarks using MediaPipe
mp_face_mesh = mp.solutions.face_mesh
with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS
            )
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Facial Landmarks')
        plt.show()

# Step 6: Evaluate model performance
print("Evaluating MediaPipe...")
ground_truth = np.array([[100, 100], [150, 150], [200, 200], [250, 250], [300, 300]])
time_taken, nme, robustness = evaluate_model_performance("MediaPipe", detect_with_mediapipe, image, ground_truth)

print("\nPerformance Summary:")
print(f"MediaPipe Detection Time: {time_taken:.4f} seconds")
print(f"MediaPipe Normalized Mean Error (NME): {nme}")
print(f"MediaPipe Robustness Score: {robustness} landmarks")
