# prompt: create a succesfully code that can make RGB color of image processing by the uploaded image

from google.colab import files
import matplotlib.pyplot as plt
import cv2
import numpy as np

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

  # Split the image into its BGR color channels
  b, g, r = cv2.split(img)

  # Display the original and individual color channels
  plt.figure(figsize=(15, 5))

  plt.subplot(1, 4, 1)
  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
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
