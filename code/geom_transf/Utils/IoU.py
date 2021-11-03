import cv2
import numpy
import matplotlib.pyplot as plt


img = cv2.imread("../LogoGoogle.jpg")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

first_bb_points = [[250, 210], [440, 210], [440, 390], [250, 390]]
stencil = numpy.zeros(img.shape).astype(img.dtype)
contours = [numpy.array(first_bb_points)]
color = [255, 255, 255]
cv2.fillPoly(stencil, contours, color)
result1 = cv2.bitwise_and(img, stencil)
result1 = cv2.cvtColor(result1, cv2.COLOR_BGR2RGB)
plt.imshow(result1)

second_bb_points = [[280, 190], [438, 190], [438, 390], [280, 390]]
stencil = numpy.zeros(img.shape).astype(img.dtype)
contours = [numpy.array(second_bb_points)]
color = [255, 255, 255]
cv2.fillPoly(stencil, contours, color)
result2 = cv2.bitwise_and(img, stencil)
result2 = cv2.cvtColor(result2, cv2.COLOR_BGR2RGB)
plt.imshow(result2)

intersection = numpy.logical_and(result1, result2)
union = numpy.logical_or(result1, result2)
iou_score = numpy.sum(intersection) / numpy.sum(union)
print("IoU is %s" %iou_score)