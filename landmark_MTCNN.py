# prompt: from the uploaded image, use another model MTCNN model  for identify the facial landmark and annotate it ton the face

!pip install mtcnn

from mtcnn import MTCNN
import cv2
from google.colab.patches import cv2_imshow

try:
    image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    detector = MTCNN()
    faces = detector.detect_faces(image)

    for face in faces:
        x, y, width, height = face['box']
        keypoints = face['keypoints']

        # Draw bounding box around the face
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Annotate facial landmarks
        for keypoint_name, (x_coord, y_coord) in keypoints.items():
            cv2.circle(image, (x_coord, y_coord), 2, (0, 0, 255), 2)
            cv2.putText(image, keypoint_name, (x_coord, y_coord), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2_imshow(image)

except Exception as e:
    print(f"Error processing the image with MTCNN: {e}")
