import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate intensity frequencies
def calculate_intensity_frequencies(image):
    # Flatten the grayscale image to a 1D array of pixel intensities
    flattened_image = image.flatten()
    
    # Calculate frequencies of each intensity (0-255)
    intensity_frequencies = np.bincount(flattened_image, minlength=256)
    
    return intensity_frequencies

# Load the image
image_path = 'cat.jpg'  # Replace with your image file path
image = cv2.imread(image_path)

# Convert to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Image",grayscale_image)
# Calculate intensity frequencies
intensity_frequencies = calculate_intensity_frequencies(grayscale_image)

# Display the frequency of each intensity
plt.bar(range(256), intensity_frequencies, color='gray')
plt.title('Frequency of Each Intensity')
plt.xlabel('Pixel Intensity (0-255)')
plt.ylabel('Frequency')
plt.show()

# Save the grayscale image
cv2.imwrite('grayscale_image.jpg', grayscale_image)

print("Intensity frequencies:", intensity_frequencies)
