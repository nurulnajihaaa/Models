# grayscale + mediapipe

from google.colab import files
import matplotlib.pyplot as plt
import cv2
import numpy as np
import mediapipe as mp
from google.colab.patches import cv2_imshow

# Step 1: Upload Image
uploaded = files.upload()

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

for fn in uploaded.keys():
    print(f'User uploaded file "{fn}" with length {len(uploaded[fn])} bytes')

    # Read the image using OpenCV
    img = cv2.imdecode(np.frombuffer(uploaded[fn], np.uint8), cv2.IMREAD_COLOR)

    # Check if the image was loaded successfully
    if img is None:
        print(f"Error: Could not read image file '{fn}'. Please ensure it's a valid image format.")
        continue

    # Step 2: Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 3: Process Facial Landmark with MediaPipe on the original (color) image
    try:
        with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for MediaPipe
            results = face_mesh.process(image_rgb)

            # Annotate facial landmarks if detected
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=img,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)

                # Display the annotated image
                plt.figure(figsize=(10, 5))

                plt.subplot(1, 2, 1)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Original Image
                plt.title('Original Image with Facial Landmarks')

                plt.subplot(1, 2, 2)
                plt.imshow(gray_img, cmap='gray')  # Grayscale Image
                plt.title('Grayscale Image')

                plt.show()

            else:
                print("No faces detected in the image.")
    except Exception as e:
        print(f"Error processing the image: {e}")
