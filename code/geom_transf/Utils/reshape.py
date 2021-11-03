
import cv2

# Input image

input = cv2.imread('C:\\Users\\Eros Bignardi\\projectCV\\data\\train\\cloth\\12129734_1.jpg')

# Get input size

height, width = input.shape[:2]

# Desired "pixelated" size

w, h = (192, 256)

# Resize input to "pixelated" size

temp = cv2.resize(input, (w, h), interpolation=cv2.INTER_LINEAR)

# Initialize output image
output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
print(output.shape)

temp = cv2.resize(input, (w, h))

cv2.imwrite("C:\\Users\\Eros Bignardi\\projectCV\\result.jpg", temp)