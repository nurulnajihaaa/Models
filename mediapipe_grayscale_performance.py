# Install dependencies
!pip install mediapipe mtcnn dlib opencv-python-headless

# Import libraries
import time
import cv2
import numpy as np
import mediapipe as mp
from google.colab import files
import matplotlib.pyplot as plt

# Function to calculate Normalized Mean Error (NME)
def calculate_nme(detected_landmarks, ground_truth_landmarks):
    if detected_landmarks.shape != ground_truth_landmarks.shape:
        raise ValueError("Detected and ground truth landmarks must have the same shape.")
    distances = np.linalg.norm(detected_landmarks - ground_truth_landmarks, axis=1)
    ipd = np.linalg.norm(ground_truth_landmarks[0] - ground_truth_landmarks[1])  # Interpupillary distance
    nme = np.mean(distances) / ipd
    return nme

# Function to calculate Robustness Score
def calculate_robustness(detected_landmarks, ground_truth_landmarks, threshold=5):
    distances = np.linalg.norm(detected_landmarks - ground_truth_landmarks, axis=1)
    robustness = np.mean(distances < threshold) * 100
    return robustness

# Upload image and convert to grayscale
print("Upload an image to process:")
uploaded = files.upload()

for filename in uploaded.keys():
    print(f"Processing file: {filename}")
    # Read the image
    image = cv2.imdecode(np.frombuffer(uploaded[filename], np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        print(f"Error: Could not read image file '{filename}'. Please ensure it's a valid image format.")
        continue

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Display original and grayscale image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(gray_image, cmap='gray')
    plt.title("Grayscale Image")
    plt.show()

    # MediaPipe Face Mesh setup
    print("Processing image with MediaPipe...")
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        start_time = time.time()
        results = face_mesh.process(rgb_image)
        detection_time = time.time() - start_time

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = np.array([
                    [lmk.x * rgb_image.shape[1], lmk.y * rgb_image.shape[0]]
                    for lmk in face_landmarks.landmark
                ])

            # Display landmarks on the image
            annotated_image = rgb_image.copy()
            for (x, y) in landmarks.astype(int):
                cv2.circle(annotated_image, (x, y), 1, (0, 255, 0), -1)
            plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            plt.title("MediaPipe Detected Landmarks")
            plt.axis("off")
            plt.show()

            # Ground truth landmarks (placeholder, replace with actual values if available)
            ground_truth_landmarks = np.array([
                [150, 200], [180, 210], [160, 240], [140, 280], [190, 290]
            ])

            # Evaluate performance
            try:
                nme = calculate_nme(landmarks[:5], ground_truth_landmarks)
                robustness = calculate_robustness(landmarks[:5], ground_truth_landmarks)
                print("\nPerformance Metrics:")
                print(f"Detection Time: {detection_time:.4f} seconds")
                print(f"Normalized Mean Error (NME): {nme:.4f}")
                print(f"Robustness Score: {robustness:.2f}%")
            except Exception as e:
                print(f"Error during performance evaluation: {e}")
        else:
            print("No faces detected in the image.")
