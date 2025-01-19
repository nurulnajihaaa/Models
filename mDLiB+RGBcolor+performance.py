# Install necessary libraries
!pip install dlib opencv-python-headless matplotlib

# Import libraries
from google.colab import files
import cv2
import numpy as np
import dlib
import matplotlib.pyplot as plt
import time

# Function to upload and process an image
def upload_and_process_image():
    uploaded = files.upload()
    for fn in uploaded.keys():
        print(f'User uploaded file "{fn}" with length {len(uploaded[fn])} bytes')
        # Read the image
        image = cv2.imdecode(np.frombuffer(uploaded[fn], np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Error: Could not read the image. Please ensure it is a valid format.")
        return image

# Download and prepare the Dlib predictor
!wget -q http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
!bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
predictor_path = "shape_predictor_68_face_landmarks.dat"

# Function to detect facial landmarks using Dlib
def detect_facial_landmarks(image, predictor_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        print("No faces detected.")
        return image, None

    # Annotate landmarks on the image
    for face in faces:
        landmarks = predictor(gray, face)
        for n in range(0, 68):
            x, y = landmarks.part(n).x, landmarks.part(n).y
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    return image, landmarks

# Function to process and display RGB channels
def display_rgb_channels(image):
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

# Function to calculate Normalized Mean Error (NME)
def calculate_nme(detected_landmarks, ground_truth_landmarks):
    if detected_landmarks is None:
        return None
    detected = np.array([(p.x, p.y) for p in detected_landmarks.parts()[:5]])
    distances = np.linalg.norm(detected - ground_truth_landmarks, axis=1)
    ipd = np.linalg.norm(ground_truth_landmarks[0] - ground_truth_landmarks[1])  # Interpupillary distance
    nme = np.mean(distances) / ipd
    return nme

# Ground truth landmarks for evaluation (example values)
ground_truth_landmarks = np.array([
    [130, 210], [170, 210], [150, 250], [140, 290], [180, 290]
])

# Main script
try:
    # Step 1: Upload image
    print("Please upload an image...")
    image = upload_and_process_image()

    # Step 2: Facial landmark detection
    print("\nDetecting facial landmarks using Dlib...")
    start_time = time.time()
    annotated_image, landmarks = detect_facial_landmarks(image.copy(), predictor_path)
    detection_time = time.time() - start_time

    if landmarks:
        print(f"Dlib Detection Time: {detection_time:.4f} seconds")
        nme = calculate_nme(landmarks, ground_truth_landmarks)
        print(f"Dlib Normalized Mean Error (NME): {nme:.4f}" if nme else "Dlib NME could not be calculated.")
        robustness_score = len(landmarks.parts())
        print(f"Dlib Robustness Score: {robustness_score} landmarks detected")
    else:
        print("Dlib did not detect any landmarks.")

    # Display the annotated image
    cv2.imshow("Dlib Annotated Image", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Step 3: Display RGB channels
    print("\nDisplaying RGB color channels...")
    display_rgb_channels(image)

except Exception as e:
    print(f"An error occurred: {e}")
