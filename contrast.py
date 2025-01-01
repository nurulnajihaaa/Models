# prompt: create a succesfully code that can make contrast of image processing by the uploaded image

from google.colab import files
import matplotlib.pyplot as plt
import cv2
import numpy as np

def adjust_contrast(image, contrast_factor):
    """Adjusts the contrast of an image.

    Args:
        image: The input image as a NumPy array.
        contrast_factor: A float representing the contrast adjustment factor. 
                          Values > 1 increase contrast, values < 1 decrease it.

    Returns:
        The contrast-adjusted image as a NumPy array.
    """

    # Ensure the contrast factor is within a reasonable range
    contrast_factor = max(0.0, min(contrast_factor, 3.0)) # Limit to prevent extreme values

    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)
    return adjusted_image

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
    
    # Adjust contrast (example: increase contrast by a factor of 1.5)
    contrast_factor = 1.5  # Example contrast factor, modify as needed
    enhanced_image = adjust_contrast(img, contrast_factor)

    # Display the original and contrast-enhanced images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
    plt.title(f'Enhanced Image (Contrast Factor: {contrast_factor})')

    plt.show()
