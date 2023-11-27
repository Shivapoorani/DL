import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("D:\Shiva\DL\images.jpg")
ground_truth_mask = cv2.imread("D:\Shiva\DL\images.jpg", cv2.IMREAD_GRAYSCALE)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

sure_bg = cv2.dilate(opening, kernel, iterations=3)

dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

cv2.watershed(image, markers)
image[markers == -1] = [0, 0, 255]

intersection = np.logical_and(ground_truth_mask, markers == -1)
union = np.logical_or(ground_truth_mask, markers == -1)
iou = np.sum(intersection) / np.sum(union)

print("Intersection over Union (IoU):", iou)

plt.subplot(131), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Segmentation Result')
plt.subplot(132), plt.imshow(ground_truth_mask, cmap='gray'), plt.title('Ground Truth Mask')
plt.subplot(133), plt.imshow(markers, cmap='jet'), plt.title('Watershed Markers')
plt.show()
