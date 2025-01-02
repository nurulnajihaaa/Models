# ada tiga2 performance

# Install dependencies
!pip install mediapipe mtcnn dlib opencv-python-headless

# Import libraries
import time
import numpy as np
import cv2
import mediapipe as mp
from mtcnn import MTCNN
import dlib
from google.colab import files

# Upload the image file
print("Please upload an image file...")
uploaded = files.upload()
image_path = list(uploaded.keys())[0]  # Get the uploaded file's name
image_array = cv2.imread(image_path)  # Load the image

if image_array is None:
    print("Error: Could not load the image. Please upload a valid image.")
    exit()

# Resize the image to avoid memory issues
image_array = cv2.resize(image_array, (640, 480))

# Download and extract Dlib's shape predictor
!wget -q http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
!bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

# Dlib's predictor path
predictor_path = "/content/shape_predictor_68_face_landmarks.dat"

# Function to calculate Normalized Mean Error (NME)
def calculate_nme(detected_landmarks, ground_truth_landmarks):
    if detected_landmarks.shape != ground_truth_landmarks.shape:
        raise ValueError("Detected and ground truth landmarks must have the same shape.")
    distances = np.linalg.norm(detected_landmarks - ground_truth_landmarks, axis=1)
    ipd = np.linalg.norm(ground_truth_landmarks[0] - ground_truth_landmarks[1])  # Interpupillary distance
    nme = np.mean(distances) / ipd
    return nme

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
            return np.array(landmarks[:5])  # Using the first 5 landmarks as ground truth example
    return None

# Detection function for MTCNN
def detect_with_mtcnn(image):
    detector = MTCNN()
    results = detector.detect_faces(image)
    if len(results) == 0:
        return None
    keypoints = results[0]['keypoints']
    landmarks = [
        keypoints['left_eye'], keypoints['right_eye'], keypoints['nose'],
        keypoints['mouth_left'], keypoints['mouth_right']
    ]
    return np.array(landmarks)

# Detection function for Dlib
def detect_with_dlib(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    landmarks = predictor(gray, faces[0])
    return np.array([(p.x, p.y) for p in landmarks.parts()[:5]])  # Using the first 5 landmarks

# Ground truth landmarks for each model
ground_truth_mediapipe = np.array([
    [150, 200], [180, 210], [160, 240], [140, 280], [190, 290]
])

ground_truth_mtcnn = np.array([
    [100, 200], [200, 200], [150, 250], [120, 300], [180, 300]
])

ground_truth_dlib = np.array([
    [130, 210], [170, 210], [150, 250], [140, 290], [180, 290]
])

# Function to evaluate the model performance
def evaluate_model_performance(model_name, detection_function, image_array, ground_truth_landmarks):
    start_time = time.time()
    robustness_score = 0
    try:
        detected_landmarks = detection_function(image_array.copy())
        elapsed_time = time.time() - start_time
        print(f"{model_name} Detection Time: {elapsed_time:.4f} seconds")

        if detected_landmarks is not None:
            nme = calculate_nme(detected_landmarks, ground_truth_landmarks)
            robustness_score = len(detected_landmarks)
            print(f"{model_name} Normalized Mean Error (NME): {nme:.4f}")
            print(f"{model_name} Robustness Score (Detected Landmarks): {robustness_score}")
        else:
            print(f"{model_name}: No landmarks detected.")
    except Exception as e:
        print(f"Error during {model_name} detection: {e}")
    return elapsed_time, robustness_score

# Evaluate MediaPipe performance
print("\nEvaluating MediaPipe...")
mediapipe_time, mediapipe_robustness = evaluate_model_performance(
    "MediaPipe", detect_with_mediapipe, image_array, ground_truth_mediapipe
)

# Evaluate MTCNN performance
print("\nEvaluating MTCNN...")
mtcnn_time, mtcnn_robustness = evaluate_model_performance(
    "MTCNN", detect_with_mtcnn, image_array, ground_truth_mtcnn
)

# Evaluate Dlib performance
print("\nEvaluating Dlib...")
dlib_time, dlib_robustness = evaluate_model_performance(
    "Dlib", detect_with_dlib, image_array, ground_truth_dlib
)

# Print performance summary
print("\nPerformance Summary:")
print(f"MediaPipe Time: {mediapipe_time:.4f} seconds, Robustness: {mediapipe_robustness} landmarks")
print(f"MTCNN Time: {mtcnn_time:.4f} seconds, Robustness: {mtcnn_robustness} landmarks")
print(f"Dlib Time: {dlib_time:.4f} seconds, Robustness: {dlib_robustness} landmarks")
