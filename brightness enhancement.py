# prompt: create a succesfully code that can make brightness enhancement of image processing by the uploaded image

from google.colab import files
import matplotlib.pyplot as plt
import cv2
import numpy as np

def adjust_contrast(image, contrast_factor):
    """Adjusts the contrast of an image."""
    contrast_factor = max(0.0, min(contrast_factor, 3.0))
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)
    return adjusted_image

def enhance_image(image_path, contrast_factor=1.5):
    """Enhances the image contrast and returns original and enhanced images."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image file '{image_path}'.")
        return None, None
    enhanced_image = adjust_contrast(img, contrast_factor)
    return img, enhanced_image

uploaded = files.upload()

for fn in uploaded.keys():
    print(f'User uploaded file "{fn}" with length {len(uploaded[fn])} bytes')
    
    with open(fn, 'wb') as f:
        f.write(uploaded[fn])
        
    original_image, enhanced_image = enhance_image(fn)

    if original_image is not None and enhanced_image is not None:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
        plt.title('Enhanced Image (Contrast)')
        plt.show()
    
    !rm {fn}
