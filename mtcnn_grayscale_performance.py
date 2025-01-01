# Import necessary libraries
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from mtcnn import MTCNN

# Function to upload and read an image
def upload_image():
    uploaded = files.upload()
    for filename in uploaded.keys():
        print(f"Uploaded file: {filename}")
        return cv2.imread(filename)

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

# Function to process image with MTCNN
def detect_with_mtcnn(image):
    detector = MTCNN()
    start_time = time.time()
    results = detector.detect_faces(image)
    detection_time = time.time() - start_time

    if len(results) == 0:
        return None, detection_time
    
    keypoints = results[0]['keypoints']
    landmarks = np.array([
        keypoints['left_eye'], keypoints['right_eye'], keypoints['nose'],
        keypoints['mouth_left'], keypoints['mouth_right']
    ])
    return landmarks, detection_time

# Main function
def main():
    # Step 1: Upload image
    print("Upload an image to process:")
    image = upload_image()
    
    if image is None:
        print("Error: Could not load the image.")
        return

    # Step 2: Convert to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(gray_img, cmap='gray')
    plt.title("Grayscale Image")
    plt.show()

    # Step 3: Process with MTCNN
    print("Processing image with MTCNN...")
    gray_img_rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    detected_landmarks, detection_time = detect_with_mtcnn(gray_img_rgb)
    
    if detected_landmarks is None:
        print("No faces detected using MTCNN.")
        return

    # Step 4: Display MTCNN output
    for (x, y) in detected_landmarks:
        cv2.circle(gray_img_rgb, (x, y), 2, (0, 255, 0), -1)
    plt.imshow(cv2.cvtColor(gray_img_rgb, cv2.COLOR_BGR2RGB))
    plt.title("MTCNN Detected Landmarks")
    plt.axis("off")
    plt.show()

    # Step 5: Evaluate performance
    # Placeholder ground truth landmarks (replace with actual values if available)
    ground_truth_landmarks = np.array([
        [150, 200], [180, 210], [160, 240], [140, 280], [190, 290]
    ])

    try:
        nme = calculate_nme(detected_landmarks, ground_truth_landmarks)
        robustness = calculate_robustness(detected_landmarks, ground_truth_landmarks)
        print("\nPerformance Metrics:")
        print(f"Detection Time: {detection_time:.4f} seconds")
        print(f"Normalized Mean Error (NME): {nme:.4f}")
        print(f"Robustness Score: {robustness:.2f}%")
    except Exception as e:
        print(f"Error during performance evaluation: {e}")

# Run the main function
if __name__ == "__main__":
    main()
