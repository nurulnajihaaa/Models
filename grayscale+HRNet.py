# grayscale + HRNet

from google.colab import files
import matplotlib.pyplot as plt
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import dlib
from PIL import Image
import io

# Download the shape predictor model if it doesn't exist
predictor_model = "shape_predictor_68_face_landmarks.dat"
try:
    predictor = dlib.shape_predictor(predictor_model)
except RuntimeError:
    !wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    !bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
    predictor = dlib.shape_predictor(predictor_model)

# Step 1: Upload Image
uploaded = files.upload()

# Step 2: Read and process the image
for fn in uploaded.keys():
    print(f'User uploaded file "{fn}" with length {len(uploaded[fn])} bytes')

    # Read the image using OpenCV
    img = cv2.imdecode(np.frombuffer(uploaded[fn], np.uint8), cv2.IMREAD_COLOR)

    # Check if the image was loaded successfully
    if img is None:
        print(f"Error: Could not read image file '{fn}'. Please ensure it's a valid image format.")
        continue

    # Step 3: Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 4: Process Facial Landmark Detection using dlib
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale for dlib
        detector = dlib.get_frontal_face_detector()
        faces = detector(gray)

        # Annotate facial landmarks
        for face in faces:
            landmarks = predictor(gray, face)
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

        # Display the annotated image with facial landmarks
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Annotated image
        plt.title('Image with Facial Landmarks')

        plt.subplot(1, 2, 2)
        plt.imshow(gray_img, cmap='gray')  # Grayscale Image
        plt.title('Grayscale Image')

        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")
