# prompt: create a succesfully code that can make enhancement of image processing by the uploaded image

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

def enhance_image(image_path, contrast_factor=1.5):
    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if img is None:
        print(f"Error: Could not read image file '{image_path}'. Please ensure it's a valid image format.")
        return None

    # Adjust contrast
    enhanced_image = adjust_contrast(img, contrast_factor)

    return img, enhanced_image


uploaded = files.upload()

for fn in uploaded.keys():
    print('User uploaded file "{name}" with length {length} bytes'.format(
        name=fn, length=len(uploaded[fn])))
    
    # Save the uploaded file temporarily
    with open(fn, 'wb') as f:
        f.write(uploaded[fn])
        
    original_image, enhanced_image = enhance_image(fn)

    if original_image is not None and enhanced_image is not None:
        # Display the original and contrast-enhanced images
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Enhanced Image (Contrast)')

        plt.show()
    
    #Optional: Remove the temporary file
    !rm {fn}
