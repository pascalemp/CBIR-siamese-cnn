import cv2
import numpy as np
from matplotlib import pyplot as plt

#Crop test code for an image of dimension 128x128

img = cv2.imread('image_0002.jpg',0)
edges = cv2.Canny(img,100,200)

quadrant_1 = []
quadrant_2 = []
quadrant_3 = []
quadrant_4 = []

for i in range(0,64):
    quadrant_1.append(edges[i][0:64])
    quadrant_4.append(edges[i][64:])

for i in range(64,128):
    quadrant_2.append(edges[i][0:64])
    quadrant_3.append(edges[i][64:])

amount_of_points = [0,0,0,0]

for line in quadrant_1:
    amount_of_points[0] += np.count_nonzero(line)
for line in quadrant_2:
     amount_of_points[1] += np.count_nonzero(line)
for line in quadrant_3:
     amount_of_points[2] += np.count_nonzero(line)
for line in quadrant_4:
     amount_of_points[3] += np.count_nonzero(line)

plt.subplot(131),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

if amount_of_points.index(max(amount_of_points)) == 0:
    #plt.subplot(133),plt.imshow(quadrant_1,cmap = 'gray')
    crop_img = img[0:64, 0:64]
    plt.subplot(133),plt.imshow(crop_img,cmap = 'gray')
    plt.title('Best Quadrant: 1'), plt.xticks([]), plt.yticks([])
elif amount_of_points.index(max(amount_of_points)) == 1:
    plt.subplot(133),plt.imshow(quadrant_2,cmap = 'gray')
    plt.title('Best Quadrant: 2'), plt.xticks([]), plt.yticks([])
elif amount_of_points.index(max(amount_of_points)) == 2:
    plt.subplot(133),plt.imshow(quadrant_3,cmap = 'gray')
    plt.title('Best Quadrant: 3'), plt.xticks([]), plt.yticks([])
elif amount_of_points.index(max(amount_of_points)) == 3:
    plt.subplot(133),plt.imshow(quadrant_4,cmap = 'gray')
    plt.title('Best Quadrant: 4'), plt.xticks([]), plt.yticks([])

plt.subplot(132),plt.imshow(edges,cmap = 'gray')
plt.title('All Edges'), plt.xticks([]), plt.yticks([])

plt.show()

