import numpy as np
import cv2 as cv

# Read the image
image = cv.imread("test2.jpeg")

# Convert to grayscale
grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Get height and width of the grayscale image
height, width = grayImage.shape

# Calculate half the height and width for dividing the image into 4 parts
rh = height // 2
cw = width // 2

# Create a new blank image of the same size as the grayscale image
newImage = np.zeros_like(grayImage)

# Loop over the 2x2 quadrants
for r in range(2):
    for c in range(2):
        
        # Define start and end points for the current quadrant
        startX = c * cw
        startY = r * rh
        endX = startX + cw
        endY = startY + rh
        
        # Extract the current quadrant
        section = grayImage[startY:endY, startX:endX]
        
        # Apply histogram equalization to the section
        equalized = cv.equalizeHist(section)
        
        # Place the equalized section back into the new image
        newImage[startY:endY, startX:endX] = equalized

# Display the original grayscale and the equalized image
cv.imshow("Original Grayscale Image", grayImage)
cv.imshow("Histogram Equalized Image", newImage)

# Wait for a key press and close the windows
cv.waitKey(0)
cv.destroyAllWindows()
