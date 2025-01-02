# Import necessary libraries
from google.colab import files
import matplotlib.pyplot as plt
import cv2
import numpy as np
from mtcnn import MTCNN
from google.colab.patches import cv2_imshow

# Upload an image file
uploaded = files.upload()

for fn in uploaded.keys():
    print('User uploaded file "{name}" with length {length} bytes'.format(
        name=fn, length=len(uploaded[fn])))

    # Read the image using OpenCV
    img = cv2.imdecode(np.frombuffer(uploaded[fn], np.uint8), cv2.IMREAD_COLOR)

    # Check if the image was loaded successfully
    if img is None:
        print(f"Error: Could not read image file '{fn}'. Please ensure it's a valid image format.")
        continue

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Display the original and grayscale images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Show BGR image in RGB format
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(gray_img, cmap='gray')
    plt.title('Grayscale Image')
    plt.show()

    # Process the image with MTCNN
    try:
        # Convert grayscale image back to 3-channel for MTCNN processing
        gray_img_rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        detector = MTCNN()
        faces = detector.detect_faces(gray_img_rgb)

        # Annotate faces and facial landmarks
        for face in faces:
            x, y, width, height = face['box']
            keypoints = face['keypoints']

            # Draw bounding box around the face
            cv2.rectangle(gray_img_rgb, (x, y), (x + width, y + height), (0, 255, 0), 2)

            # Annotate facial landmarks
            for keypoint_name, (x_coord, y_coord) in keypoints.items():
                cv2.circle(gray_img_rgb, (x_coord, y_coord), 2, (0, 0, 255), 2)
                cv2.putText(gray_img_rgb, keypoint_name, (x_coord, y_coord), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display the annotated image
        print("Annotated image with facial landmarks:")
        cv2_imshow(gray_img_rgb)

    except Exception as e:
        print(f"Error processing the image with MTCNN: {e}")
