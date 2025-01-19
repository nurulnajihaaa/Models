# Install dependencies
!pip install mtcnn mediapipe dlib opencv-python-headless

# Import libraries
import time
import numpy as np
import cv2
import mediapipe as mp
from mtcnn import MTCNN
import dlib
import matplotlib.pyplot as plt
from google.colab import files

# Function to adjust contrast
def adjust_contrast(image, contrast_factor):
    """Adjusts the contrast of an image."""
    contrast_factor = max(0.0, min(contrast_factor, 3.0)) # Limit to prevent extreme values
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)
    return adjusted_image

# Upload the image file
print("Please upload an image file...")
uploaded = files.upload()
image_path = list(uploaded.keys())[0]  # Get the uploaded file's name
image_array = cv2.imread(image_path)  # Load the image

if image_array is None:
    print("Error: Could not load the image. Please upload a valid image.")
    exit()

# Resize the image to avoid memory issues
image_array_resized = cv2.resize(image_array, (640, 480))

# Apply contrast adjustment
contrast_factor = 1.5  # Example contrast factor
enhanced_image = adjust_contrast(image_array_resized, contrast_factor)

# MTCNN facial landmark detection
detector = MTCNN()
faces = detector.detect_faces(image_array_resized)

# Annotate facial landmarks
for face in faces:
    x, y, width, height = face['box']
    keypoints = face['keypoints']

    # Draw bounding box around the face
    cv2.rectangle(enhanced_image, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # Annotate facial landmarks
    for keypoint_name, (x_coord, y_coord) in keypoints.items():
        cv2.circle(enhanced_image, (x_coord, y_coord), 2, (0, 0, 255), 2)
        cv2.putText(enhanced_image, keypoint_name, (x_coord, y_coord), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Display the enhanced image with MTCNN landmarks
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image_array_resized, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
plt.title(f'Enhanced Image (Contrast Factor: {contrast_factor})')

plt.show()

# Download and extract Dlib's shape predictor
!wget -q http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
!bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
predictor_path = "/content/shape_predictor_68_face_landmarks.dat"

# Function to calculate Normalized Mean Error (NME)
def calculate_nme(detected_landmarks, ground_truth_landmarks):
    if detected_landmarks.shape != ground_truth_landmarks.shape:
        raise ValueError("Detected and ground truth landmarks must have the same shape.")
    distances = np.linalg.norm(detected_landmarks - ground_truth_landmarks, axis=1)
    ipd = np.linalg.norm(ground_truth_landmarks[0] - ground_truth_landmarks[1])  # Interpupillary distance
    nme = np.mean(distances) / ipd
    return nme

# Detection function for MTCNN
def detect_with_mtcnn(image):
    results = detector.detect_faces(image)
    if len(results) == 0:
        return None
    keypoints = results[0]['keypoints']
    landmarks = [
        keypoints['left_eye'], keypoints['right_eye'], keypoints['nose'],
        keypoints['mouth_left'], keypoints['mouth_right']
    ]
    return np.array(landmarks)

# Ground truth landmarks for MTCNN
ground_truth_mtcnn = np.array([
    [100, 200], [200, 200], [150, 250], [120, 300], [180, 300]
])

# Function to evaluate the model performance
def evaluate_model_performance(model_name, detection_function, image_array, ground_truth_landmarks):
    start_time = time.time()
    try:
        detected_landmarks = detection_function(image_array.copy())
        elapsed_time = time.time() - start_time
        print(f"{model_name} Detection Time: {elapsed_time:.4f} seconds")

        if detected_landmarks is not None:
            nme = calculate_nme(detected_landmarks, ground_truth_landmarks)
            print(f"{model_name} Normalized Mean Error (NME): {nme:.4f}")
        else:
            print(f"{model_name}: No landmarks detected.")
    except Exception as e:
        print(f"Error during {model_name} detection: {e}")
    return elapsed_time

# Evaluate MTCNN performance
print("\nEvaluating MTCNN...")
mtcnn_time = evaluate_model_performance(
    "MTCNN", detect_with_mtcnn, enhanced_image, ground_truth_mtcnn
)

# Print performance summary
print("\nPerformance Summary:")
print(f"MTCNN Time: {mtcnn_time:.4f} seconds")
