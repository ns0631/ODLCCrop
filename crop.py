import os, cv2, time
import numpy as np

base = cv2.imread('output.png', 0)

while 1:
    filepath = input('Enter image path: ')
    if not os.path.exists(filepath):
        print('Invalid filename.')
    else:
        break

start = time.time()
image = cv2.imread(filepath).copy()
for row in range(image.shape[0]):
    for column in range(image.shape[1]):
        if base[row, column] == 0:
            image[row, column, 0] = 0
            image[row, column, 1] = 0
            image[row, column, 2] = 0

cv2.imwrite('cropped.png', image)
end = time.time()
print('Written to cropped.png. Time:', end - start)
