# prompt: from the uploaded image, identify the facial landmark and annotate it on the face

!pip install mediapipe

import cv2
import mediapipe as mp
from google.colab.patches import cv2_imshow
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


# Assuming image_array is already defined from your previous code
try:
    with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

            cv2_imshow(image)
        else:
            print("No faces detected in the image.")

except Exception as e:
    print(f"Error processing the image: {e}")
